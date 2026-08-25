//! Upstream/provider plumbing shared by the API-server proxy and the agent loop:
//! message normalization, model->upstream resolution, OpenAI chat-completion
//! calls, and MCP tool collection/execution. Lifted verbatim from
//! `core/server/proxy.rs` (no behavior change) so both the server path and
//! `core/agent/loop.rs` consume one implementation.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use futures_util::StreamExt;
use reqwest::Client;
use rmcp::model::{CallToolRequestParam, CallToolResult};
#[cfg(not(feature = "cli"))]
use tauri_plugin_llamacpp::state::LlamacppState;
use tokio::sync::{mpsc, Mutex};

use crate::core::agent::events::StreamEvent;
use crate::core::openai_schema::{
    http_status_indicates_api_key_retry, normalize_openai_tool_parameters_schema,
};
#[cfg(not(feature = "cli"))]
use crate::core::server::proxy::router_upstream;
#[cfg(not(feature = "cli"))]
use crate::core::server::MlxBackendSession;
use crate::core::{
    mcp::models::McpSettings,
    mcp::truncate::truncate_tool_result,
    state::{ProviderConfig, SharedMcpServers},
};

fn assistant_json_path(jan_data_folder: &str, assistant_id: &str) -> PathBuf {
    PathBuf::from(jan_data_folder)
        .join("assistants")
        .join(assistant_id)
        .join("assistant.json")
}

pub(crate) fn load_assistant_config(
    jan_data_folder: &str,
    assistant_id: &str,
) -> Result<(Option<String>, Option<String>), String> {
    let assistant_path = assistant_json_path(jan_data_folder, assistant_id);
    let raw = fs::read_to_string(&assistant_path)
        .map_err(|e| format!("Failed to read assistant.json: {assistant_path:?}: {e}"))?;

    let parsed: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("Invalid assistant.json ({assistant_id}): {e}"))?;

    let instructions = parsed
        .get("instructions")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let model = parsed
        .get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok((instructions, model))
}

pub(crate) fn parse_openai_messages(
    messages: &serde_json::Value,
) -> Result<Vec<serde_json::Value>, String> {
    let arr = messages
        .as_array()
        .ok_or("Request body must include 'messages' as an array")?;

    let mut out = Vec::with_capacity(arr.len());
    for msg in arr {
        let role = msg
            .get("role")
            .and_then(|v| v.as_str())
            .ok_or("Each message must include a string 'role'")?;

        // Assistant tool-call turns carry `tool_calls` and may have `content: null`
        // (or omit it entirely) per the OpenAI protocol. `tool` result messages
        // carry a `tool_call_id`. These shapes flow back into the conversation
        // history (see MessagesUpdated), so a follow-up request re-submits them and
        // must preserve them verbatim -- otherwise the assistant/tool pairing is
        // broken and content-null turns are wrongly rejected.
        let has_tool_calls = msg
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());
        let tool_call_id = msg.get("tool_call_id").and_then(|v| v.as_str());

        // Content is a plain string or an OpenAI multimodal content-part array
        // (text + image_url); pass either through verbatim. It may be null (or
        // absent) only when the assistant message carries tool_calls.
        let content = match msg.get("content") {
            Some(v @ serde_json::Value::String(_)) | Some(v @ serde_json::Value::Array(_)) => {
                v.clone()
            }
            // Null/absent content is valid for an assistant turn whose payload is
            // entirely tool calls; normalize to null so upstream sees a well-formed
            // message.
            Some(serde_json::Value::Null) | None if has_tool_calls || tool_call_id.is_some() => {
                serde_json::Value::Null
            }
            _ => return Err("Each message must include 'content' as a string or array".into()),
        };

        let mut obj = serde_json::Map::new();
        obj.insert("role".to_string(), serde_json::json!(role));
        obj.insert("content".to_string(), content);
        // Preserve a resent assistant turn's reasoning alongside its content and
        // tool calls. Some providers require prior reasoning to be resubmitted
        // to keep a (local llama.cpp `preserve_thinking`) chat template honest:
        // dropping it would shrink earlier assistant turns and force reprocessing
        // of the KV-cache prefix. Only assistant messages carry it; user/tool/
        // system passes are untouched by construction.
        if role == "assistant" {
            if let Some(r) = msg.get("reasoning_content").and_then(|v| v.as_str()) {
                if !r.is_empty() {
                    obj.insert("reasoning_content".to_string(), serde_json::json!(r));
                }
            }
        }
        if has_tool_calls {
            obj.insert("tool_calls".to_string(), msg["tool_calls"].clone());
        }
        if let Some(id) = tool_call_id {
            obj.insert("tool_call_id".to_string(), serde_json::json!(id));
        }
        out.push(serde_json::Value::Object(obj));
    }
    Ok(out)
}

/// Repairs a "dangling" tool-call turn: an assistant message with
/// `tool_calls` whose ids don't all have a matching `role: "tool"` reply
/// immediately after. Anthropic (and some other providers) reject the whole
/// request outright when this happens, rather than just the offending turn.
///
/// This can happen if a previous run was interrupted before a tool result was
/// ever recorded -- e.g. the process crashed or was force-killed while an
/// `ask`/permission prompt was still pending -- and the incomplete turn was
/// then persisted and later resumed/replayed. Insert a synthetic error result
/// for each missing id so the conversation is always well-formed by the time
/// it leaves this process, regardless of how the gap was introduced. Returns
/// the number of ids repaired.
pub(crate) fn repair_dangling_tool_calls(messages: &mut Vec<serde_json::Value>) -> usize {
    let mut repaired = 0;
    let mut i = 0;
    while i < messages.len() {
        let ids: Vec<String> = messages[i]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|tc| tc.get("id").and_then(|v| v.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if ids.is_empty() {
            i += 1;
            continue;
        }
        // A tool-call turn's replies are the run of `role: "tool"` messages
        // immediately following it; the run ends at the next message that
        // isn't a tool reply.
        let mut answered: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut j = i + 1;
        while j < messages.len()
            && messages[j].get("role").and_then(|v| v.as_str()) == Some("tool")
        {
            if let Some(id) = messages[j].get("tool_call_id").and_then(|v| v.as_str()) {
                answered.insert(id);
            }
            j += 1;
        }
        let missing: Vec<&String> = ids.iter().filter(|id| !answered.contains(id.as_str())).collect();
        for id in &missing {
            messages.insert(
                j,
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": "ERROR: this tool call was interrupted before it produced a \
                        result (e.g. an unanswered question or a cancelled run); treat it as \
                        failed.",
                }),
            );
            j += 1;
        }
        repaired += missing.len();
        i = j;
    }
    repaired
}

/// Drops "poisoned" tool calls: an assistant `tool_calls` entry whose
/// `function.arguments` is not parsable JSON. A model (observed with
/// DeepSeek/vLLM) can end a stream mid-argument while still reporting
/// `finish_reason: "tool_calls"`, so the truncated call is persisted into the
/// thread. Every later turn resends it, and an OpenAI-compatible upstream
/// rejects the whole request with 422 -- the session is wedged, because the
/// poison is in the history the agent keeps replaying.
///
/// Removal, not reconstruction: a truncated argument cannot be recovered, and
/// inventing one would run a tool the model never actually asked for. The call
/// is dropped along with any `role: "tool"` reply carrying its `tool_call_id`,
/// so no orphaned result is left behind. Valid sibling calls in the same turn
/// survive; an assistant turn whose calls are ALL dropped keeps its text and
/// loses only the `tool_calls` key (and is removed entirely if that leaves it
/// empty, which would otherwise be a contentless assistant turn some providers
/// reject). Returns the number of calls dropped.
///
/// Runs before [`repair_dangling_tool_calls`], so a surviving call that lost
/// its result still gets the synthetic error reply from that pass.
pub(crate) fn drop_malformed_tool_calls(messages: &mut Vec<serde_json::Value>) -> usize {
    // A call is poison when arguments are present but unparsable. An absent or
    // empty `arguments` is the well-formed "no arguments" spelling several
    // providers use, and is left alone.
    fn is_malformed(call: &serde_json::Value) -> bool {
        let Some(args) = call
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
        else {
            return false;
        };
        let args = args.trim();
        !args.is_empty() && serde_json::from_str::<serde_json::Value>(args).is_err()
    }

    let mut dropped_ids: Vec<String> = Vec::new();
    let mut dropped = 0;
    for msg in messages.iter_mut() {
        let Some(calls) = msg.get("tool_calls").and_then(|v| v.as_array()) else {
            continue;
        };
        if !calls.iter().any(is_malformed) {
            continue;
        }
        let mut kept: Vec<serde_json::Value> = Vec::with_capacity(calls.len());
        for call in calls {
            if is_malformed(call) {
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    dropped_ids.push(id.to_string());
                }
                dropped += 1;
            } else {
                kept.push(call.clone());
            }
        }
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        if kept.is_empty() {
            obj.remove("tool_calls");
        } else {
            obj.insert("tool_calls".to_string(), serde_json::Value::Array(kept));
        }
    }
    if dropped == 0 {
        return 0;
    }
    // Drop the results that answered a dropped call, then any assistant turn
    // left with neither text nor calls (content-null with no tool_calls is not
    // a valid turn to resend).
    messages.retain(|m| {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or_default();
        if role == "tool" {
            let id = m
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            return !dropped_ids.iter().any(|d| d == id);
        }
        if role != "assistant" || m.get("tool_calls").is_some() {
            return true;
        }
        // Keep a turn that still says something; a content-null/empty one is
        // now an empty shell left by the dropped call.
        match m.get("content") {
            Some(serde_json::Value::String(s)) => !s.trim().is_empty(),
            Some(serde_json::Value::Array(a)) => !a.is_empty(),
            _ => false,
        }
    });
    dropped
}

pub(crate) fn set_system_prompt(messages: &mut Vec<serde_json::Value>, system_prompt: &str) {
    messages.retain(|m| m.get("role").and_then(|r| r.as_str()) != Some("system"));
    messages.insert(
        0,
        serde_json::json!({
            "role": "system",
            "content": system_prompt
        }),
    );
}

pub(crate) fn extract_tool_calls(response: &serde_json::Value) -> Vec<serde_json::Value> {
    response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|choices| choices.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
        .map(|arr| arr.to_vec())
        .unwrap_or_default()
}

pub(crate) fn extract_choice_message(response: &serde_json::Value) -> Option<&serde_json::Value> {
    response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|choices| choices.first())
        .and_then(|c| c.get("message"))
}

fn mcp_call_result_to_string(result: &CallToolResult) -> String {
    let parts: Vec<String> = result
        .content
        .iter()
        .filter_map(|c| c.as_text())
        .map(|t| t.text.clone())
        .collect();

    if result.is_error == Some(true) {
        if parts.is_empty() {
            "ERROR".to_string()
        } else {
            format!("ERROR: {}", parts.join("\n"))
        }
    } else {
        parts.join("\n")
    }
}

/// Resolve `model_id` to an upstream URL + key chain. The desktop build also
/// resolves local engines (MLX session, llama-server router); the `cli` build is
/// remote-only, so a model with no provider entry is unresolvable.
pub(crate) async fn resolve_upstream_for_model(
    model_id: &str,
    provider_configs: Arc<Mutex<HashMap<String, ProviderConfig>>>,
    #[cfg(not(feature = "cli"))] llama_state: Arc<LlamacppState>,
    #[cfg(not(feature = "cli"))] mlx_sessions: Arc<Mutex<HashMap<i32, MlxBackendSession>>>,
) -> Result<(String, Vec<String>), String> {
    let destination_path = "/chat/completions";

    let pc = provider_configs.lock().await;
    let offers = |config: &ProviderConfig| config.models.iter().any(|m| m == model_id);

    // The same model id can be listed by several providers -- typically a local
    // engine descriptor (no base_url) plus a Jan desktop API server exposing the
    // same model over HTTP. `HashMap` iteration order is randomized, so pick the
    // usable one deterministically instead of whichever hashes first. Only the
    // CLI needs this: the desktop resolves base_url-less providers through its
    // in-process engine branches below.
    #[cfg(feature = "cli")]
    let first_match = pc
        .iter()
        .filter(|(_, config)| {
            config.base_url.as_deref().is_some_and(|u| !u.is_empty()) && offers(config)
        })
        .min_by_key(|(name, _)| name.to_string())
        .or_else(|| pc.iter().find(|(_, config)| offers(config)));
    #[cfg(not(feature = "cli"))]
    let first_match = pc.iter().find(|(_, config)| offers(config));

    let provider_name = first_match
        .map(|(_, config)| config.provider.clone())
        .or_else(|| {
            if let Some(sep_pos) = model_id.find('/') {
                let potential_provider: &str = &model_id[..sep_pos];
                if pc.contains_key(potential_provider) {
                    return Some(potential_provider.to_string());
                }
            }
            pc.get(model_id).map(|c| c.provider.clone())
        });
    drop(pc);

    if let Some(provider) = provider_name {
        let pc2 = provider_configs.lock().await;
        if let Some(provider_cfg) = pc2.get(provider.as_str()).cloned() {
            // A populated base_url means an HTTP upstream (cloud, or a local
            // engine whose live endpoint was registered at runtime). A local
            // engine loaded from persisted settings has none -- fall through to
            // the MLX session / llama-server router resolution below.
            if let Some(api_url) = provider_cfg.base_url.clone().filter(|u| !u.is_empty()) {
                let url = format!("{}{}", api_url, destination_path);
                return Ok((url, provider_cfg.bearer_key_chain()));
            }
        }
    }

    #[cfg(not(feature = "cli"))]
    {
        let mlx_guard = mlx_sessions.lock().await;
        if let Some(info) = mlx_guard.values().find(|s| s.info.model_id == model_id) {
            let target_port = info.info.port;
            return Ok((
                format!("http://127.0.0.1:{target_port}/v1{destination_path}"),
                vec![info.info.api_key.clone()],
            ));
        }
        drop(mlx_guard);

        if let Some((url, key)) = router_upstream(&llama_state, destination_path).await {
            return Ok((url, vec![key]));
        }
    }

    Err(format!("No upstream session found for model '{model_id}'"))
}

pub(crate) fn copy_optional_chat_params(
    from: &serde_json::Value,
    into: &mut serde_json::Map<String, serde_json::Value>,
) {
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "stop_sequences",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
    ] {
        if let Some(v) = from.get(key) {
            into.insert(key.to_string(), v.clone());
        }
    }
}

pub(crate) async fn collect_mcp_openai_tools(
    mcp_servers: &SharedMcpServers,
    mcp_settings: &Arc<Mutex<McpSettings>>,
) -> Result<(Vec<serde_json::Value>, HashMap<String, String>), String> {
    let timeout_duration = mcp_settings.lock().await.tool_call_timeout_duration();
    let servers = mcp_servers.lock().await;

    let mut openai_tools = Vec::new();
    let mut tool_to_server: HashMap<String, String> = HashMap::new();

    // Probe every server concurrently so one slow/hanging server can't serialize
    // the whole collection behind its timeout (previously each server waited out
    // the full timeout before the next was contacted).
    let listings = futures_util::future::join_all(servers.iter().map(|(server_name, service)| {
        async move {
            let result = match tokio::time::timeout(timeout_duration, service.list_all_tools()).await
            {
                Ok(Ok(tools)) => Some(tools),
                Ok(Err(e)) => {
                    log::warn!("MCP server {} failed to list tools: {}", server_name, e);
                    None
                }
                Err(_) => {
                    log::warn!(
                        "Listing MCP tools timed out after {} seconds on server {}",
                        timeout_duration.as_secs(),
                        server_name
                    );
                    None
                }
            };
            (server_name.clone(), result)
        }
    }))
    .await;

    for (server_name, tools) in listings {
        let Some(tools) = tools else { continue };
        for tool in tools {
            tool_to_server.insert(tool.name.to_string(), server_name.clone());

            // Normalize schemas before sending them to strict OpenAI-compatible providers.
            // The `get_tools` Tauri command still returns raw schemas; the frontend
            // normalizes those separately before provider registration.
            let mut parameters = serde_json::Value::Object((*tool.input_schema).clone());
            normalize_openai_tool_parameters_schema(&mut parameters);
            let description = tool
                .description
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default();

            openai_tools.push(serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": description,
                    "parameters": parameters
                }
            }));
        }
    }

    Ok((openai_tools, tool_to_server))
}

pub(crate) async fn execute_mcp_tool_calls(
    tool_calls: &[serde_json::Value],
    tool_to_server: &HashMap<String, String>,
    mcp_servers: &SharedMcpServers,
    mcp_settings: &Arc<Mutex<McpSettings>>,
) -> Vec<(String, String)> {
    let (timeout_duration, tool_output_cap) = {
        let settings = mcp_settings.lock().await;
        (
            settings.tool_call_timeout_duration(),
            settings.tool_output_cap(None),
        )
    };
    let servers = mcp_servers.lock().await;

    let mut results = Vec::with_capacity(tool_calls.len());

    for tc in tool_calls {
        let tool_call_id = tc
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tool_name = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let args_str = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
            .unwrap_or("{}");

        let args_value: serde_json::Value =
            serde_json::from_str(args_str).unwrap_or_else(|_| serde_json::json!({}));

        let args_map: serde_json::Map<String, serde_json::Value> =
            if let Some(obj) = args_value.as_object() {
                obj.clone()
            } else {
                serde_json::Map::new()
            };

        let Some(server_name) = tool_to_server.get(&tool_name) else {
            results.push((
                tool_call_id,
                format!("ERROR: No MCP server registered for tool '{tool_name}'"),
            ));
            continue;
        };

        let Some(service) = servers.get(server_name) else {
            results.push((
                tool_call_id,
                format!("ERROR: MCP server '{server_name}' not found in runtime state"),
            ));
            continue;
        };

        let tool_call = service.call_tool(CallToolRequestParam {
            name: tool_name.clone().into(),
            arguments: Some(args_map),
        });

        let result = match tokio::time::timeout(timeout_duration, tool_call).await {
            Ok(call_result) => call_result.map_err(|e| e.to_string()),
            Err(_) => Err(format!(
                "Tool call '{tool_name}' timed out after {} seconds",
                timeout_duration.as_secs()
            )),
        };

        let tool_result_string = match result {
            // Same cap as the desktop path: this string is appended to the agent's
            // message history, so an unbounded result would blow the context here too.
            Ok(res) => mcp_call_result_to_string(&truncate_tool_result(&res, tool_output_cap)),
            Err(e) => format!("ERROR: {e}"),
        };

        results.push((tool_call_id, tool_result_string));
    }

    results
}

#[cfg(not(feature = "cli"))]
pub(crate) async fn call_openai_chat_completions(
    client: &Client,
    upstream_url: &str,
    api_keys: &[String],
    body: &serde_json::Value,
    retry: RetryPolicy,
) -> Result<serde_json::Value, String> {
    // The desktop proxy hands this completion back to an OpenAI-compatible
    // client (the Jan API server), not to the agent loop. So it deliberately
    // does NOT get the answerless-completion retry below: turning an
    // empty-but-valid 200 into an error would change that surface's contract
    // (the client asked for a completion and got one). The agent loop never
    // calls this path anyway - it always streams. Transport/status retries
    // still apply, because a dropped connect or a 429 is indistinguishable
    // from the streaming path's and nothing has been committed to any client
    // at that point.
    let attempts: Vec<Option<&str>> = if api_keys.is_empty() {
        vec![None]
    } else {
        api_keys.iter().map(|s| Some(s.as_str())).collect()
    };

    let mut last_err = String::new();
    let mut retries_used: u32 = 0;
    loop {
        // Retry-After belongs to this attempt's last response; a break driven
        // by a send error produced no new response, so a stale value from an
        // earlier attempt must not leak into this retry's delay.
        let mut retry_after: Option<std::time::Duration> = None;
        for (i, key_ref) in attempts.iter().enumerate() {
            let mut req = client
                .post(upstream_url)
                .header("Content-Type", "application/json")
                .header("Accept-Encoding", "identity");

            if let Some(key) = key_ref {
                req = req.header("Authorization", format!("Bearer {key}"));
            }

            let resp = match send_once(req.body(body.to_string()), retry.causes).await {
                Ok(resp) => resp,
                Err((desc, retryable)) => {
                    // A send error means no response arrived, so nothing was
                    // handed to the client and replaying is safe. A non-retryable
                    // send error (bytes already flowed: is_body/is_decode, or a
                    // deterministic is_builder) is fatal on the spot.
                    if retryable {
                        last_err = desc;
                        break;
                    }
                    return Err(desc);
                }
            };

            let status = resp.status();
            if !status.is_success() {
                // Headers are borrowed from the response; text() consumes it, so
                // capture Retry-After before reading the body.
                retry_after = retry_after_delay(resp.headers());
                let text = resp.text().await.map_err(|e| {
                    format!(
                        "Reading the upstream response failed ({upstream_url}): {}",
                        describe_request_error(&e)
                    )
                })?;
                last_err = format!("Upstream returned HTTP {status}: {text}");
                if is_context_overflow_body(&text) {
                    // Fatal for the same reason as the streaming path: an
                    // overflow is deterministic, so resending the identical
                    // bytes cannot succeed and would spend the whole budget
                    // failing. The caller reacts to the marker instead.
                    last_err = format!("[{CONTEXT_OVERFLOW_MARKER}] {last_err}");
                    return Err(last_err);
                }
                if http_status_indicates_api_key_retry(status) && i + 1 < attempts.len() {
                    log::warn!("OpenAI completion: HTTP {status} with API key index {i}, trying next key");
                    continue;
                }
                if is_retryable_status(status, retry.causes) {
                    break;
                }
                return Err(last_err);
            }

            let text = resp.text().await.map_err(|e| {
                format!(
                    "Reading the upstream response failed ({upstream_url}): {}",
                    describe_request_error(&e)
                )
            })?;
            return serde_json::from_str::<serde_json::Value>(&text)
                .map_err(|e| format!("Failed to parse upstream JSON: {e}. Body: {text}"));
        }

        if !retry.allows(retries_used) {
            return Err(retry_exhausted_message(&last_err, retries_used));
        }
        let delay = retry.delay(retries_used, retry_after);
        log::warn!(
            "OpenAI completion: {last_err}; retrying in {}ms (attempt {retries_used}/{})",
            delay.as_millis(),
            retry.max_retries
        );
        tokio::time::sleep(delay).await;
        retries_used += 1;
    }
}

/// Streaming counterpart of [`call_openai_chat_completions`]. Forces `stream:true`
/// (with usage), emits `StreamEvent::Token` per content delta, and reconstructs
/// an OpenAI non-streaming completion JSON so the rest of the loop (tool-call
/// extraction, history append) is identical to the non-streaming path.
/// Stable marker prefixed onto an upstream error when the failure looks like a
/// context/prompt-length overflow, so the agent loop can react (compact + retry)
/// instead of surfacing the raw provider error. Works uniformly for local
/// (llama-server) and remote (OpenAI/Anthropic/Google) since all report overflow
/// via an HTTP error body rather than the streamed deltas.
pub(crate) const CONTEXT_OVERFLOW_MARKER: &str = "context-overflow";

/// True when a provider error body reads like a context/prompt-length overflow.
/// Matches the OpenAI `context_length_exceeded` code, Anthropic's "prompt is too
/// long", Google's token-limit phrasing, and llama-server's context messages.
pub(crate) fn is_context_overflow_body(body: &str) -> bool {
    let b = body.to_lowercase();
    b.contains("context_length_exceeded")
        || b.contains("maximum context length")
        || b.contains("prompt is too long")
        || b.contains("exceeds the maximum number of tokens")
        || b.contains("the request exceeds the available context")
        || b.contains("exceeds the available context size")
        || b.contains("exceed_context_size_error")
        || b.contains("exceed context")
        || b.contains("context window")
        || (b.contains("context") && b.contains("too long"))
}

/// True when an error string carries the [`CONTEXT_OVERFLOW_MARKER`].
pub(crate) fn is_context_overflow_error(err: &str) -> bool {
    err.contains(CONTEXT_OVERFLOW_MARKER)
}

/// True when the upstream rejected the request *because* an assistant turn
/// carries `reasoning_content`. The field is a DeepSeek extension that some
/// providers require to be resent (llama.cpp `preserve_thinking`) while strict
/// OpenAI-compatible endpoints (Groq, vLLM's pydantic validation) reject the
/// whole request rather than ignoring the unknown key. Recognizing it lets the
/// caller drop the field and retry instead of failing the turn, so
/// `send_reasoning` does not have to be configured per provider by hand.
/// Requires both the field name and a rejection phrase: a body that merely
/// echoes the request must not be read as a rejection of it.
pub(crate) fn is_reasoning_field_error(err: &str) -> bool {
    let e = err.to_lowercase();
    e.contains("reasoning_content")
        && [
            "unsupported",
            "unrecognized",
            "unknown",
            "not permitted",
            "not allowed",
            "unexpected",
            "additional",
            "extra input",
            "extra field",
            "invalid",
        ]
        .iter()
        .any(|phrase| e.contains(phrase))
}

/// Every message in an error's `source()` chain, outermost cause first.
/// `reqwest::Error` prints only its own layer -- `error sending request for url
/// (...)` -- so the reason the request never left (DNS failure, refused
/// connection, TLS mismatch, dropped socket) is one or more sources down and is
/// otherwise lost to the user. Consecutive duplicates are collapsed: hyper and
/// its io error often stringify identically.
fn error_source_chain(err: &dyn std::error::Error) -> Vec<String> {
    let mut chain = Vec::new();
    let mut cur = err.source();
    while let Some(e) = cur {
        let msg = e.to_string();
        if !msg.trim().is_empty() && chain.last() != Some(&msg) {
            chain.push(msg);
        }
        cur = e.source();
    }
    chain
}

/// True when a failed send can be retried safely: the request never produced a
/// response, so nothing has been streamed to the caller and no side effect on
/// the upstream is implied.
///
/// Covers a refused/failed connect and the stale-keep-alive family -- hyper
/// reports a pooled connection the peer had already closed as `connection
/// closed before message completed`, or as an `ECONNRESET`/`EPIPE` io error if
/// the RST lands while the request is going out.
///
/// A timeout counts only under [`RetryCauses::All`], and only because of how
/// this client is built: [`agent_http_client`] sets a *connect* timeout and
/// deliberately no overall request timeout, so a timeout here means the
/// connection was never established -- the request bytes never reached the
/// upstream. `is_body` and `is_decode` stay excluded under every policy: those
/// mean bytes already flowed, so a replay could duplicate work the upstream
/// already did. `is_builder` is a deterministic local mistake that a resend
/// cannot fix.
fn is_retryable_send_error(err: &reqwest::Error, causes: RetryCauses) -> bool {
    if err.is_body() || err.is_decode() || err.is_builder() {
        return false;
    }
    if err.is_timeout() {
        return causes == RetryCauses::All;
    }
    err.is_connect() || chain_indicates_dropped_connection(&error_source_chain(err))
}

/// Whether an error's cause chain names a connection the peer dropped. Matched
/// on text because the io error is several opaque layers down (hyper's
/// `SendRequest` -> `connection error` -> `std::io::Error`) and its `ErrorKind`
/// is not exposed through `reqwest`.
fn chain_indicates_dropped_connection(chain: &[String]) -> bool {
    const MARKERS: &[&str] = &[
        "connection closed before message completed",
        "connection reset by peer",
        "broken pipe",
        "connection aborted",
        "unexpected eof",
    ];
    chain
        .iter()
        .any(|msg| {
            let msg = msg.to_lowercase();
            MARKERS.iter().any(|m| msg.contains(m))
        })
}

/// HTTP statuses worth resending the identical request for. All three are
/// "come back later" signals from a healthy-but-busy path -- rate limiting and
/// the load-balancer family -- and all arrive as response *headers*, before a
/// single SSE byte is consumed, so nothing has been committed when one lands.
///
/// Deliberately narrow: a 500 is usually a deterministic provider bug that
/// replays identically, and a 4xx other than 429 is the request's own fault.
///
/// Never retried under [`RetryCauses::DroppedConnectionOnly`], which predates
/// any status-driven retry.
fn is_retryable_status(status: reqwest::StatusCode, causes: RetryCauses) -> bool {
    causes == RetryCauses::All && matches!(status.as_u16(), 429 | 502 | 503 | 504)
}

/// The upstream's own `Retry-After`, when it names one. Both wire forms are
/// accepted: delay-seconds (overwhelmingly what OpenAI-compatible upstreams
/// send) and the fractional-seconds some gateways emit. An HTTP-date is not
/// parsed -- it needs a clock-skew-tolerant date parser for a form this traffic
/// does not use -- and falls back to computed backoff.
///
/// Capped: a provider asking for an hour must not strand the run in an
/// invisible sleep, and the cap keeps the wait inside what a user will sit
/// through while still respecting the signal's intent.
fn retry_after_delay(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    const MAX_HONOURED: u64 = 60_000;
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let secs: f64 = raw.trim().parse().ok()?;
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    let ms = (secs * 1000.0).round() as u64;
    Some(std::time::Duration::from_millis(ms.min(MAX_HONOURED)))
}

/// Which failure classes a policy resends for.
///
/// The broad set is a CLI/TUI capability, not a global one: it ships together
/// with the `[agent]` settings that size it (`max_retries`,
/// `retry_backoff_ms`) and the TUI line that makes each attempt visible. The
/// desktop app and the OpenAI-compatible proxy expose neither, so they keep
/// the narrow pre-#8781 behaviour rather than silently inheriting a
/// ten-attempt loop their users can neither see nor turn off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetryCauses {
    /// Every cause proven safe to resend: a dropped or refused connection, a
    /// request timeout, HTTP 429/502/503/504, and an answerless 200.
    All,
    /// Only a connection that failed or was dropped before any response
    /// arrived -- exactly what `send_with_one_retry` covered before #8781.
    /// Timeouts, statuses, and answerless completions stay fatal.
    DroppedConnectionOnly,
}

/// How many times a failed attempt is resent, how long the waits between them
/// grow, and which failures qualify. `max_retries` and `backoff_ms` are the
/// resolved form of the user-facing `[agent].max_retries` and
/// `[agent].retry_backoff_ms`; `causes` is set by the surface, not the user.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryPolicy {
    /// Attempts *after* the first failure. `0` disables retrying entirely,
    /// which is a supported setting and not the same as leaving the key unset
    /// (unset resolves to [`DEFAULT_MAX_RETRIES`]).
    pub max_retries: u32,
    /// First backoff, doubled per attempt and jittered. `0` retries with no
    /// wait beyond the jitter floor.
    pub backoff_ms: u64,
    /// Private so a caller cannot widen its own retry surface by accident:
    /// pick a constructor that names the intent instead.
    causes: RetryCauses,
}

/// Attempts after the first failure when `[agent].max_retries` is unset. High
/// enough to ride out a provider's rate-limit window or a load balancer
/// recycling a backend, which is the failure this exists for; the run stays
/// interruptible throughout, so the ceiling is not a commitment to wait.
pub(crate) const DEFAULT_MAX_RETRIES: u32 = 10;

/// First backoff when `[agent].retry_backoff_ms` is unset. Long enough for a
/// just-recycled backend to come back, short enough that the user does not read
/// the first retry as a hang.
pub(crate) const DEFAULT_RETRY_BACKOFF_MS: u64 = 250;

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::configurable(DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BACKOFF_MS)
    }
}

impl RetryPolicy {
    /// The policy the CLI resolves from `[agent]`: every safe cause retried,
    /// with the user's own attempt count and base delay.
    pub(crate) fn configurable(max_retries: u32, backoff_ms: u64) -> Self {
        Self {
            max_retries,
            backoff_ms,
            causes: RetryCauses::All,
        }
    }

    /// The pre-#8781 behaviour, for surfaces with no retry settings and no
    /// place to show an attempt counter: one resend, only for a connection
    /// that never delivered a response. The 250ms base is now jittered to
    /// `[125ms, 250ms]` instead of fixed, which is the one deliberate
    /// difference -- it costs nothing at a single attempt and desynchronises
    /// many clients retrying one provider at once.
    pub(crate) fn legacy() -> Self {
        Self {
            max_retries: 1,
            backoff_ms: DEFAULT_RETRY_BACKOFF_MS,
            causes: RetryCauses::DroppedConnectionOnly,
        }
    }

    /// Whether a blank-but-successful completion is resent, and -- for the
    /// loop's own backstop -- whether such a completion fails the turn instead
    /// of being handed back as an empty success. Both follow the same rule:
    /// only the surfaces that opted into the #8781 behaviour get either.
    pub(crate) fn retries_answerless(&self) -> bool {
        self.causes == RetryCauses::All
    }

    /// Whether an attempt that has already failed `retries_used` times gets
    /// another go.
    fn allows(&self, retries_used: u32) -> bool {
        retries_used < self.max_retries
    }

    /// Backoff before retry number `retries_used + 1`: the base delay doubled
    /// per prior retry, capped, then jittered.
    ///
    /// Jitter is full-range (`[0.5x, 1.0x]` of the computed delay) because the
    /// failure this rides out is frequently shared -- several agents hitting one
    /// rate-limited provider retry in lockstep otherwise, and a synchronised
    /// herd re-triggers the same 429. `Retry-After`, when the upstream sends
    /// one, wins outright: it is the provider stating the actual wait.
    fn delay(&self, retries_used: u32, retry_after: Option<std::time::Duration>) -> std::time::Duration {
        if let Some(d) = retry_after {
            return d;
        }
        const MAX_BACKOFF_MS: u64 = 30_000;
        let grown = self
            .backoff_ms
            .saturating_mul(1u64 << retries_used.min(16))
            .min(MAX_BACKOFF_MS);
        let jittered = if grown == 0 {
            0
        } else {
            use rand::Rng;
            rand::thread_rng().gen_range(grown / 2..=grown)
        };
        std::time::Duration::from_millis(jittered)
    }
}

/// Send a request once, mapping a failure to a description and whether resending
/// the identical bytes is safe under `causes`. The body is a `String` for every
/// caller here, so the retry path always has a clone available; a body that
/// cannot be cloned reports as non-retryable rather than being silently
/// dropped.
async fn send_once(
    req: reqwest::RequestBuilder,
    causes: RetryCauses,
) -> Result<reqwest::Response, (String, bool)> {
    match req.send().await {
        Ok(resp) => Ok(resp),
        Err(e) => {
            let retryable = is_retryable_send_error(&e, causes);
            Err((
                format!("Upstream request failed: {}", describe_request_error(&e)),
                retryable,
            ))
        }
    }
}

/// The message an exhausted retry budget surfaces, keeping the last upstream
/// failure verbatim inside it (a test greps for `HTTP 429`). When nothing was
/// retried (`retries_used == 0`) the original failure is returned untouched --
/// the transport never gave us a second chance, so editorialising about retries
/// that did not happen would mislead the user about the likely cause.
fn retry_exhausted_message(last_err: &str, retries_used: u32) -> String {
    if retries_used == 0 {
        last_err.to_string()
    } else {
        format!("Upstream failed after {retries_used} retries; last failure: {last_err}")
    }
}

/// True when a completion is a blank turn: no tool calls, no content, and a
/// finish_reason of "stop" or absent. Such a 200 produced nothing the caller
/// can use, so resending it is safe -- but only when it is truly empty.
///
/// The content check is deliberately NOT trimmed: a whitespace-only answer DID
/// stream Token bytes, and replaying it would duplicate those tokens in the
/// transcript. "Content empty" must mean "no Token bytes were streamed at all",
/// which is exactly the condition that makes a replay leave no trace.
///
/// A "length" finish is excluded: that is a real truncation the caller detects
/// and handles separately, not a blank turn, and must not be silently retried.
///
/// Reasoning-only turns (reasoning streamed, no content) DO count as blank: the
/// user asked a question and got no answer, which is the whole defect. Reasoning
/// is display-only and never joins the assistant `content` that is resent as
/// history, so replaying one duplicates no transcript bytes.
pub(crate) fn completion_is_answerless(completion: &serde_json::Value) -> bool {
    if !extract_tool_calls(completion).is_empty() {
        return false;
    }
    // `content` is a String or Null off the streaming path, but this predicate
    // also guards the shared loop, where a completion can come from any invoker
    // -- a local engine or a proxied JSON body may carry OpenAI content-part
    // arrays. A non-empty array is a real answer; only a missing, null, or
    // empty-string content is a blank turn.
    let content_empty = match extract_choice_message(completion).and_then(|m| m.get("content")) {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::String(s)) => s.is_empty(),
        Some(serde_json::Value::Array(parts)) => parts.is_empty(),
        Some(_) => false,
    };
    if !content_empty {
        return false;
    }
    match completion
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
    {
        None | Some("stop") => true,
        Some(_) => false,
    }
}

/// The HTTP client every agent turn goes through. `Client::new()`'s defaults are
/// wrong for this traffic in one specific way: connections idle for up to 90s
/// stay in the pool, but a turn spends far longer than that running tools, and
/// the peer (or its load balancer) closes an idle connection well before then.
/// Reusing one is the `Connection reset by peer` a long session eventually hits,
/// so the pool is kept shorter-lived than any plausible upstream idle timeout.
/// No overall request timeout: a streamed answer legitimately runs for minutes.
pub(crate) fn agent_http_client() -> Client {
    Client::builder()
        .pool_idle_timeout(std::time::Duration::from_secs(15))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// Names the proxy environment variables in force, without their values (they
/// routinely carry credentials). A proxy set in the environment is a common
/// reason a request fails for Jan and for nothing else, and it is invisible in
/// the error itself.
fn proxy_env_hint() -> Option<String> {
    const VARS: &[&str] = &[
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ];
    let set: Vec<&str> = VARS
        .iter()
        .copied()
        .filter(|name| {
            std::env::var_os(name).is_some_and(|v| !v.to_string_lossy().trim().is_empty())
        })
        .collect();
    (!set.is_empty()).then(|| format!("proxy env set: {}", set.join(", ")))
}

/// A failed HTTP request described well enough to act on: what stage failed, the
/// `reqwest` message, its whole cause chain, and -- for a connect or timeout
/// failure, where the environment is usually the culprit -- which proxy
/// variables are set.
pub(crate) fn describe_request_error(err: &reqwest::Error) -> String {
    let stage = if err.is_timeout() {
        "timed out"
    } else if err.is_connect() {
        "could not connect"
    } else if err.is_redirect() {
        "too many redirects"
    } else if err.is_body() || err.is_decode() {
        "response body failed"
    } else if err.is_builder() {
        "request could not be built"
    } else {
        "send failed"
    };
    let mut msg = format!("{stage}: {err}");
    let chain = error_source_chain(err);
    if !chain.is_empty() {
        msg.push_str(&format!(" (caused by: {})", chain.join(" <- ")));
    }
    if let Some(status) = err.status() {
        msg.push_str(&format!(" [HTTP {status}]"));
    }
    if err.is_connect() || err.is_timeout() {
        if let Some(hint) = proxy_env_hint() {
            msg.push_str(&format!(" [{hint}]"));
        }
    }
    msg
}

pub(crate) async fn stream_openai_chat_completions(
    client: &Client,
    upstream_url: &str,
    api_keys: &[String],
    body: &serde_json::Value,
    events: &mpsc::UnboundedSender<StreamEvent>,
    retry: RetryPolicy,
) -> Result<serde_json::Value, String> {
    // The request body is built exactly once, before the retry loop, so every
    // attempt resends the identical bytes (req_body.to_string()) rather than a
    // rebuilt request. That, plus tool execution happening only after invoke
    // returns (in loop.rs), is what keeps resending safe: at every retry
    // decision below, no tool call has executed and no content has entered the
    // transcript. A rebuilt body could silently drift across attempts (stream
    // flags, timestamps), so replay uses the original serialization verbatim.
    let mut req_body = body.clone();
    if let Some(obj) = req_body.as_object_mut() {
        obj.insert("stream".to_string(), serde_json::json!(true));
        obj.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
    }

    let attempts: Vec<Option<&str>> = if api_keys.is_empty() {
        vec![None]
    } else {
        api_keys.iter().map(|s| Some(s.as_str())).collect()
    };

    let mut last_err = String::new();
    let mut retries_used: u32 = 0;
    loop {
        // Retry-After is scoped to one outer attempt: reset here so a value
        // from an earlier attempt can never leak into a delay driven by a send
        // error or an answerless completion, neither of which produced a new
        // response. Within a single attempt the most recent non-success
        // response wins, including across a key rotation -- a Retry-After the
        // provider just sent is the freshest signal available, whichever key
        // drew it.
        let mut retry_after: Option<std::time::Duration> = None;
        for (i, key_ref) in attempts.iter().enumerate() {
            let mut req = client
                .post(upstream_url)
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header("Accept-Encoding", "identity");

            if let Some(key) = key_ref {
                req = req.header("Authorization", format!("Bearer {key}"));
            }

            let resp = match send_once(req.body(req_body.to_string()), retry.causes).await {
                Ok(resp) => resp,
                Err((desc, retryable)) => {
                    // A send error means no response arrived at all, so nothing
                    // was streamed and a replay is safe. A non-retryable send
                    // error means bytes already flowed (is_body / is_decode) or
                    // the request could never have been sent (is_builder), so
                    // it is fatal on the spot.
                    if retryable {
                        last_err = desc;
                        break;
                    }
                    return Err(desc);
                }
            };

            let status = resp.status();
            if status.is_success() {
                match consume_openai_sse(resp, events).await {
                    Err(e) => {
                        // An Err from consume_openai_sse surfaces MID-STREAM:
                        // either the socket died after content had already
                        // streamed, or a data:{"error":...} object landed.
                        // Bytes may already be committed to the transcript, so
                        // resending could duplicate them - treat as fatal, never
                        // retry.
                        return Err(e);
                    }
                    Ok(completion)
                        if retry.retries_answerless()
                            && completion_is_answerless(&completion) =>
                    {
                        // A 200 that produced nothing - no content tokens, no
                        // tool calls, and a "stop" / absent finish - left no
                        // trace on the transcript, so replaying is safe. Empty
                        // content means no Token event was ever emitted and no
                        // tool call means loop.rs executed nothing. Reasoning
                        // deltas may already have been emitted, but reasoning is
                        // display-only and never joins the assistant `content`
                        // resent as history, so it does not block the retry. We
                        // reach the retry decision below rather than returning
                        // the blank turn the user would otherwise see.
                        let fr = completion["choices"][0]
                            .get("finish_reason")
                            .and_then(|f| f.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "absent".to_string());
                        last_err = format!(
                            "Upstream returned an empty completion (finish_reason: {fr})"
                        );
                        break;
                    }
                    Ok(completion) => return Ok(completion),
                }
            } else {
                // Headers are borrowed from the response; text() consumes it, so
                // capture Retry-After before reading the body.
                retry_after = retry_after_delay(resp.headers());
                let text = resp.text().await.unwrap_or_default();
                last_err = format!("Upstream returned HTTP {status}: {text}");
                if is_context_overflow_body(&text) {
                    last_err = format!("[{CONTEXT_OVERFLOW_MARKER}] {last_err}");
                    // A context-overflow body is fatal: loop.rs reacts to it by
                    // compacting and retrying, so resending the identical bytes
                    // here would burn the whole budget on a deterministic
                    // failure. Fail out and let the compaction path own it.
                    return Err(last_err);
                }
                if http_status_indicates_api_key_retry(status) && i + 1 < attempts.len() {
                    log::warn!("OpenAI stream: HTTP {status} with API key index {i}, trying next key");
                    continue;
                }
                if is_retryable_status(status, retry.causes) {
                    break;
                }
                return Err(last_err);
            }
        }

        if !retry.allows(retries_used) {
            return Err(retry_exhausted_message(&last_err, retries_used));
        }
        let delay = retry.delay(retries_used, retry_after);
        // Emit the wait before sleeping so it is visible rather than reading as
        // a hang, then sleep asynchronously: this runs inside the spawned run
        // task, so a blocking std::thread::sleep would stall the TUI render loop
        // (#8710).
        let _ = events.send(StreamEvent::Retrying {
            attempt: retries_used + 1,
            max: retry.max_retries,
            delay_ms: delay.as_millis() as u64,
            reason: last_err.clone(),
        });
        log::warn!(
            "OpenAI stream: {last_err}; retrying in {}ms (attempt {}/{}",
            delay.as_millis(),
            retries_used + 1,
            retry.max_retries
        );
        tokio::time::sleep(delay).await;
        retries_used += 1;
    }
}

#[derive(Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
    /// A `ToolCallStarted` was already emitted for this call; guards the
    /// once-per-call in-progress signal against later argument deltas.
    started_emitted: bool,
    /// Bytes of `arguments` already forwarded as `ToolCallArgsDelta`. Zero
    /// until the first forward, which replays whatever arrived before the call
    /// could be announced.
    args_forwarded: usize,
}

/// Accumulates OpenAI SSE deltas into a single reconstructed completion. Kept
/// separate from the byte-stream reader so it is unit-testable without a live
/// HTTP response.
#[derive(Default)]
struct SseAccumulator {
    content: String,
    /// Natively streamed reasoning (`reasoning_content` deltas), accumulated so
    /// the reconstructed completion carries it back on the assistant message.
    /// Kept apart from `content`: reasoning is never part of the answer prose,
    /// but a caller resending assistant turns may forward it to the model.
    reasoning: String,
    tool_calls: Vec<ToolCallAccum>,
    finish_reason: Option<String>,
    usage: Option<serde_json::Value>,
    /// An error object delivered inside the stream (`data: {"error": {...}}`).
    /// OpenAI-compatible upstreams can fail mid-stream after a `200 OK`; without
    /// capturing it the run would end as a silent "no answer" instead of
    /// surfacing the failure. Propagated as an `Err` by `consume_openai_sse`.
    error: Option<String>,
}

impl SseAccumulator {
    /// Parse one raw SSE line (`data: {...}`); non-`data:`/blank lines are ignored.
    fn ingest_line(&mut self, line: &str, events: &mpsc::UnboundedSender<StreamEvent>) {
        if let Some(rest) = line.trim_end_matches('\r').strip_prefix("data:") {
            let data = rest.trim();
            if !data.is_empty() {
                self.ingest(data, events);
            }
        }
    }

    fn ingest(&mut self, data: &str, events: &mpsc::UnboundedSender<StreamEvent>) {
        if data == "[DONE]" {
            return;
        }
        let json: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return,
        };

        if let Some(err) = json.get("error").filter(|e| !e.is_null()) {
            let message = err
                .get("message")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| err.to_string());
            let kind = err.get("type").and_then(|v| v.as_str());
            self.error = Some(match kind {
                Some(t) if !t.is_empty() => format!("{t}: {message}"),
                _ => message,
            });
            return;
        }

        if let Some(u) = json.get("usage") {
            if !u.is_null() {
                self.usage = Some(u.clone());
            }
        }

        let Some(choice) = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
        else {
            return;
        };

        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            self.finish_reason = Some(fr.to_string());
        }

        let Some(delta) = choice.get("delta") else {
            return;
        };

        // Native reasoning: providers exposing a dedicated `reasoning_content`
        // field stream it as `Reasoning` events, never as content tokens, so
        // consumers get the boundary for free instead of re-parsing synthetic
        // `<think>` tags. Providers that inline tags in `content` still flow
        // through `Token`; consumers keep the tag-stripping fallback for them.
        if let Some(text) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                self.reasoning.push_str(text);
                let _ = events.send(StreamEvent::Reasoning {
                    text: text.to_string(),
                });
            }
        }

        if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                self.content.push_str(text);
                let _ = events.send(StreamEvent::Token {
                    text: text.to_string(),
                });
            }
        }

        if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                while self.tool_calls.len() <= idx {
                    self.tool_calls.push(ToolCallAccum::default());
                }
                let slot = &mut self.tool_calls[idx];
                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    if !id.is_empty() {
                        slot.id = id.to_string();
                    }
                }
                let func = tc.get("function");
                if let Some(name) = func
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .filter(|n| !n.is_empty())
                {
                    slot.name = name.to_string();
                }
                let arg_delta = func
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty());
                if let Some(arg) = arg_delta {
                    slot.arguments.push_str(arg);
                }

                // Signal the in-progress tool call the instant both id and name
                // are known, so consumers can show activity while arguments are
                // still streaming (a long window for large write/edit calls).
                if !slot.started_emitted && !slot.id.is_empty() && !slot.name.is_empty() {
                    slot.started_emitted = true;
                    let _ = events.send(StreamEvent::ToolCallStarted {
                        id: slot.id.clone(),
                        name: slot.name.clone(),
                    });
                }

                // Forward the argument text itself, but only once the call has
                // been announced -- a delta with no preceding `ToolCallStarted`
                // has no call for the consumer to attach it to. Providers that
                // send arguments before the id/name are covered by the replay
                // below, which flushes what was buffered before the announce.
                if slot.started_emitted {
                    if let Some(arg) = arg_delta {
                        let delta = if slot.args_forwarded == 0 {
                            // First forward after the announce: ship everything
                            // accumulated so far, including any chunks that
                            // arrived before id/name were known.
                            slot.arguments.clone()
                        } else {
                            arg.to_string()
                        };
                        slot.args_forwarded = slot.arguments.len();
                        let _ = events.send(StreamEvent::ToolCallArgsDelta {
                            id: slot.id.clone(),
                            delta,
                        });
                    }
                }
            }
        }
    }

    fn into_completion(self) -> serde_json::Value {
        let tool_calls: Vec<serde_json::Value> = self
            .tool_calls
            .into_iter()
            .filter(|t| !t.id.is_empty() || !t.name.is_empty() || !t.arguments.is_empty())
            .map(|t| {
                serde_json::json!({
                    "id": t.id,
                    "type": "function",
                    "function": { "name": t.name, "arguments": t.arguments }
                })
            })
            .collect();

        let mut message = serde_json::Map::new();
        message.insert("role".to_string(), serde_json::json!("assistant"));
        message.insert(
            "content".to_string(),
            if self.content.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::json!(self.content)
            },
        );
        // Reasoning stays out of `content` but is carried on the message so a
        // caller that resends assistant turns to the model can forward it (and
        // so a turn that only reasoned still surfaces its reasoning). Empty is
        // omitted so non-reasoning providers produce an unchanged shape.
        if !self.reasoning.is_empty() {
            message.insert("reasoning_content".to_string(), serde_json::json!(self.reasoning));
        }
        if !tool_calls.is_empty() {
            message.insert(
                "tool_calls".to_string(),
                serde_json::Value::Array(tool_calls),
            );
        }

        let mut choice = serde_json::Map::new();
        choice.insert("index".to_string(), serde_json::json!(0));
        choice.insert("message".to_string(), serde_json::Value::Object(message));
        choice.insert(
            "finish_reason".to_string(),
            self.finish_reason
                .map(|s| serde_json::json!(s))
                .unwrap_or(serde_json::Value::Null),
        );

        let mut completion = serde_json::Map::new();
        completion.insert(
            "choices".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::Object(choice)]),
        );
        if let Some(u) = self.usage {
            completion.insert("usage".to_string(), u);
        }

        serde_json::Value::Object(completion)
    }
}

/// Drain every complete (newline-terminated) line from `buf` into `acc`,
/// leaving any trailing partial line buffered for the next chunk.
fn drain_complete_lines(
    buf: &mut String,
    acc: &mut SseAccumulator,
    events: &mpsc::UnboundedSender<StreamEvent>,
) {
    while let Some(nl) = buf.find('\n') {
        let line = buf[..nl].to_string();
        buf.drain(..=nl);
        acc.ingest_line(&line, events);
    }
}

/// Ingest a final, non-newline-terminated line left after the stream closes.
/// Providers may end with `data: {...}` and no trailing blank line / `[DONE]`;
/// without this the last chunk's finish_reason and tool-call args are dropped.
fn flush_trailing_line(
    buf: &str,
    acc: &mut SseAccumulator,
    events: &mpsc::UnboundedSender<StreamEvent>,
) {
    if !buf.trim().is_empty() {
        acc.ingest_line(buf, events);
    }
}

async fn consume_openai_sse(
    resp: reqwest::Response,
    events: &mpsc::UnboundedSender<StreamEvent>,
) -> Result<serde_json::Value, String> {
    let url = resp.url().to_string();
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut acc = SseAccumulator::default();
    let mut bytes = 0usize;

    while let Some(chunk) = stream.next().await {
        // A stream that dies mid-response is the hardest case to tell apart from
        // an empty answer, so the byte count so far goes in the message.
        let chunk = chunk.map_err(|e| {
            format!(
                "Upstream stream error ({url}) after {bytes} bytes: {}",
                describe_request_error(&e)
            )
        })?;
        bytes += chunk.len();
        buf.push_str(&String::from_utf8_lossy(&chunk));
        drain_complete_lines(&mut buf, &mut acc, events);
    }
    flush_trailing_line(&buf, &mut acc, events);

    if let Some(err) = acc.error.take() {
        let msg = format!("Upstream stream error: {err}");
        return Err(if is_context_overflow_body(&err) {
            format!("[{CONTEXT_OVERFLOW_MARKER}] {msg}")
        } else {
            msg
        });
    }

    Ok(acc.into_completion())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sink() -> (
        mpsc::UnboundedSender<StreamEvent>,
        mpsc::UnboundedReceiver<StreamEvent>,
    ) {
        mpsc::unbounded_channel()
    }

    /// `reqwest` prints only its own layer, so the cause chain is where the
    /// actual failure lives -- the whole point of `describe_request_error`.
    #[test]
    fn error_source_chain_lists_every_cause_and_collapses_repeats() {
        #[derive(Debug)]
        struct Err2(&'static str, Option<Box<Err2>>);
        impl std::fmt::Display for Err2 {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.0)
            }
        }
        impl std::error::Error for Err2 {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.1.as_ref().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
            }
        }

        // The outermost message is printed by the caller, so the chain starts at
        // its first source; the repeated innermost layer collapses.
        let inner = Err2("connection refused", None);
        let middle = Err2("connection refused", Some(Box::new(inner)));
        let connect = Err2("tcp connect error", Some(Box::new(middle)));
        let outer = Err2("error sending request", Some(Box::new(connect)));
        assert_eq!(
            error_source_chain(&outer),
            vec![
                "tcp connect error".to_string(),
                "connection refused".to_string()
            ],
            "sources only, consecutive duplicates collapsed"
        );
        assert!(
            error_source_chain(&Err2("alone", None)).is_empty(),
            "no sources -> nothing to add"
        );
    }

    /// The reported error must name the stage and carry the OS-level reason, not
    /// just the URL. Port 1 on loopback refuses without touching the network.
    #[tokio::test]
    async fn describe_request_error_names_the_stage_and_the_os_cause() {
        let err = Client::new()
            .post("http://127.0.0.1:1/v1/chat/completions")
            .body("{}")
            .send()
            .await
            .expect_err("loopback port 1 refuses");
        let msg = describe_request_error(&err);
        assert!(msg.starts_with("could not connect: "), "stage named: {msg}");
        assert!(msg.contains("caused by: "), "cause chain present: {msg}");
        assert!(
            msg.to_lowercase().contains("refused") || msg.to_lowercase().contains("connect"),
            "the OS reason survives: {msg}"
        );
    }

    #[test]
    fn dropped_connection_is_recognised_from_the_cause_chain() {
        assert!(chain_indicates_dropped_connection(&[
            "client error (SendRequest)".to_string(),
            "connection error".to_string(),
            "Connection reset by peer (os error 104)".to_string(),
        ]));
        assert!(chain_indicates_dropped_connection(&[
            "connection closed before message completed".to_string()
        ]));
        assert!(
            !chain_indicates_dropped_connection(&[
                "dns error".to_string(),
                "failed to lookup address information".to_string()
            ]),
            "a name that does not resolve is not a dropped connection"
        );
        assert!(!chain_indicates_dropped_connection(&[]));
    }

    /// A refused connect never reached the peer, so retrying it is safe under
    /// every policy. A timeout is retryable too -- and only because of how this
    /// client is built: [`agent_http_client`] sets a *connect* timeout and no
    /// overall request timeout, so a timeout means the connection was never
    /// established and the request bytes never reached the upstream. Resending
    /// them cannot double-charge the provider, which is why a timeout is no
    /// longer excluded under [`RetryCauses::All`]. It stays excluded under
    /// `DroppedConnectionOnly`, which reproduces the pre-#8781 classification
    /// the desktop and proxy surfaces still run.
    #[tokio::test]
    async fn a_refused_connect_retries_everywhere_but_a_timeout_only_under_all_causes() {
        let refused = Client::new()
            .post("http://127.0.0.1:1/v1/chat/completions")
            .body("{}")
            .send()
            .await
            .expect_err("loopback port 1 refuses");
        assert!(
            is_retryable_send_error(&refused, RetryCauses::All),
            "{refused}"
        );
        assert!(
            is_retryable_send_error(&refused, RetryCauses::DroppedConnectionOnly),
            "a refused connect is the one cause the legacy policy retries: {refused}"
        );

        // 10.255.255.1 is a reserved address that black-holes rather than
        // refusing, so the connect attempt hits the timeout instead.
        let timed_out = Client::builder()
            .connect_timeout(std::time::Duration::from_millis(50))
            .build()
            .expect("client")
            .post("http://10.255.255.1:81/v1/chat/completions")
            .body("{}")
            .send()
            .await
            .expect_err("black-holed address times out");
        // Best-effort only: 10.255.255.1 black-holes on some hosts but refuses
        // or routes differently on others, so this may not time out at all. The
        // assertions below are kept deliberately conditional rather than
        // flaky; the real, routing-independent defence is the behavioural test
        // `a_timeout_is_retried_through_the_loop` below, which drives a timeout
        // produced by a held-open local connection through the retry loop.
        if timed_out.is_timeout() {
            assert!(
                is_retryable_send_error(&timed_out, RetryCauses::All),
                "{timed_out}"
            );
            assert!(
                !is_retryable_send_error(&timed_out, RetryCauses::DroppedConnectionOnly),
                "the legacy policy never retried a timeout: {timed_out}"
            );
        }
    }

    /// A request timeout is retried through the real loop: the first connection
    /// accepts the request and then never writes a response (a genuinely held,
    /// unanswered connection), so `reqwest` hits the client's overall request
    /// timeout; the retry lands on the second connection and streams a real
    /// answer. Deterministic and routing-independent because the timeout comes
    /// from our own [`Client::builder().timeout`], not from a route-dependent
    /// black-holed address -- the classification test above stays best-effort
    /// for that reason, and this test is the actual behavioural defence that a
    /// timeout is resent and that the resend is announced as a retry.
    #[tokio::test]
    async fn a_timeout_is_retried_through_the_loop() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        // The first connection reads the request and then holds the socket open
        // without writing a byte, so the client's request timeout fires. The
        // second connection answers with a normal stream. Each attempt gets its
        // own connection, so the timeout cannot be mistaken for an unanswered
        // pool and the retry is guaranteed to reach the second accept.
        let served = Arc::new(Mutex::new(0usize));
        let server_served = served.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut conn, _)) = listener.accept().await else {
                    break;
                };
                let served = server_served.clone();
                tokio::spawn(async move {
                    let mut scratch = [0u8; 4096];
                    let _ = conn.read(&mut scratch).await;
                    let n = {
                        let mut s = served.lock().await;
                        *s += 1;
                        *s
                    };
                    if n == 1 {
                        // Hold the connection open but reply with nothing, so
                        // the client times out waiting for a response.
                        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    } else {
                        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"after timeout\"},\"finish_reason\":\"stop\"}]}\n\n\
                                   data: [DONE]\n\n";
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                            sse.len()
                        );
                        let _ = conn.write_all(resp.as_bytes()).await;
                        let _ = conn.flush().await;
                    }
                });
            }
        });

        // A short overall request timeout forces a deterministic timeout on the
        // held-open first connection, entirely local and routing-independent.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(200))
            .build()
            .expect("client");

        let (tx, mut rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let completion = stream_openai_chat_completions(
            &client,
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(10, 0),
        )
        .await
        .expect("the timeout is retried into the second stream's answer");

        assert_eq!(
            completion["choices"][0]["message"]["content"], "after timeout",
            "the retry landed on the second connection's stream"
        );

        drop(tx);
        let mut retries = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Retrying { attempt, .. } = ev {
                retries.push(attempt);
            }
        }
        assert_eq!(retries, vec![1], "the timeout triggered exactly one retry");
        server.abort();
    }

    /// The failure a long turn invites: the peer reclaims a keep-alive
    /// connection while tools run, and the next request is written into a socket
    /// that is already gone. One retry must carry the turn through, transparently
    /// -- the caller sees a normal completion, not an error.
    #[tokio::test]
    async fn a_dropped_first_connection_is_retried_and_the_turn_succeeds() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            // First connection: read the request, then hang up without a byte of
            // response -- exactly what a reclaimed pooled connection looks like.
            let (mut first, _) = listener.accept().await.expect("first accept");
            let mut scratch = [0u8; 1024];
            let _ = first.read(&mut scratch).await;
            drop(first);

            // Second connection: a normal one-token SSE answer.
            let (mut second, _) = listener.accept().await.expect("second accept");
            let _ = second.read(&mut scratch).await;
            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                       data: [DONE]\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                sse.len()
            );
            let _ = second.write_all(resp.as_bytes()).await;
            let _ = second.flush().await;
        });

        let (tx, mut rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let completion = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(10, 0),
        )
        .await
        .expect("the retry carries the turn");

        assert_eq!(
            completion["choices"][0]["message"]["content"], "hi",
            "answer from the second connection: {completion}"
        );
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Token { text } = ev {
                tokens.push(text);
            }
        }
        assert_eq!(tokens, vec!["hi".to_string()], "streamed once, not twice");
        server.await.expect("server task");
    }
    /// Rate limiting: the first reply is a 429 with Retry-After: 0 (so the
    /// wait is a no-op), the second streams a real answer. The loop must resend
    /// the identical request and surface a `Retrying` event for attempt 1.
    #[tokio::test]
    async fn a_429_then_200_retries_and_signals_the_attempt() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        // One task per accepted connection, keyed by how many requests have
        // already come in: the first answers 429, the second streams a real
        // answer. A per-connection task (rather than sequential accept calls)
        // guarantees every request is answered, so the test can never hang on
        // a retry that reuses a pooled connection.
        let served = Arc::new(Mutex::new(0usize));
        let server_served = served.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut conn, _)) = listener.accept().await else {
                    break;
                };
                let served = server_served.clone();
                tokio::spawn(async move {
                    let mut scratch = [0u8; 4096];
                    let _ = conn.read(&mut scratch).await;
                    let n = {
                        let mut s = served.lock().await;
                        *s += 1;
                        *s
                    };
                    if n == 1 {
                        // Rate limit, asking not to wait.
                        let _ = conn
                            .write_all(
                                b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\nRetry-After: 0\r\nContent-Length: 0\r\n\r\n",
                            )
                            .await;
                    } else {
                        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n\
                                   data: [DONE]\n\n";
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                            sse.len()
                        );
                        let _ = conn.write_all(resp.as_bytes()).await;
                    }
                    let _ = conn.flush().await;
                });
            }
        });

        let (tx, mut rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let completion = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(10, 0),
        )
        .await
        .expect("the retry carries the 429 to a 200");

        assert_eq!(completion["choices"][0]["message"]["content"], "ok");

        drop(tx);
        let mut retries = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Retrying { attempt, .. } = ev {
                retries.push(attempt);
            }
        }
        assert_eq!(retries, vec![1], "one retry for the 429");
        server.abort();
    }

    /// The blank-turn defect: a 200 streams an answerless `stop` completion
    /// (empty content, no tool calls), which the old code returned as a
    /// successful turn. It must now retry instead, and the second (real) stream
    /// wins.
    #[tokio::test]
    async fn an_answerless_stop_completion_is_retried_not_returned() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        // The first connection returns the blank turn shape (empty content,
        // finish_reason "stop"); the second streams a real answer. A task per
        // connection plus `Connection: close` keeps each attempt on its own
        // accept, so the retry can never block on an unanswered pool.
        let served = Arc::new(Mutex::new(0usize));
        let server_served = served.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut conn, _)) = listener.accept().await else {
                    break;
                };
                let served = server_served.clone();
                tokio::spawn(async move {
                    let mut scratch = [0u8; 4096];
                    let _ = conn.read(&mut scratch).await;
                    let n = {
                        let mut s = served.lock().await;
                        *s += 1;
                        *s
                    };
                    let sse = if n == 1 {
                        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n"
                    } else {
                        "data: {\"choices\":[{\"delta\":{\"content\":\"real answer\"},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n"
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                        sse.len()
                    );
                    let _ = conn.write_all(resp.as_bytes()).await;
                    let _ = conn.flush().await;
                });
            }
        });

        let (tx, mut rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let completion = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(10, 0),
        )
        .await
        .expect("the empty turn is retried into a real answer");

        assert_eq!(completion["choices"][0]["message"]["content"], "real answer");

        drop(tx);
        let mut retries = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Retrying { attempt, .. } = ev {
                retries.push(attempt);
            }
        }
        assert_eq!(retries, vec![1], "the blank completion triggered one retry");
        server.abort();
    }

    /// Replay safety for a tool-call turn: a request that returned a tool call
    /// is a committed action -- the caller will execute it with real side
    /// effects. Resending it would repeat those effects, so the loop must hand
    /// the first completion back untouched and never issue a second request,
    /// even though a service that never answered with a tool call would be
    /// retried. This is the end-to-end proof that backing off a blank turn
    /// cannot have turned a side-effecting tool call into a double execution.
    #[tokio::test]
    async fn a_tool_call_turn_is_never_replayed() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        // The first request streams a tool call (empty content, finish_reason
        // "tool_calls"); any second request streams ordinary text instead. The
        // turn must never be resent, so `served` stays at 1.
        let served = Arc::new(Mutex::new(0usize));
        let server_served = served.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut conn, _)) = listener.accept().await else {
                    break;
                };
                let served = server_served.clone();
                tokio::spawn(async move {
                    let mut scratch = [0u8; 4096];
                    let _ = conn.read(&mut scratch).await;
                    let n = {
                        let mut s = served.lock().await;
                        *s += 1;
                        *s
                    };
                    let sse = if n == 1 {
                        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_scan\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
                         data: [DONE]\n\n"
                    } else {
                        "data: {\"choices\":[{\"delta\":{\"content\":\"second stream\"},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n"
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                        sse.len()
                    );
                    let _ = conn.write_all(resp.as_bytes()).await;
                    let _ = conn.flush().await;
                });
            }
        });

        let (tx, mut rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let completion = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(10, 0),
        )
        .await
        .expect("the tool call is returned, not replayed");

        assert_eq!(
            *served.lock().await, 1,
            "a tool-call turn is never resent, however many retries are allowed"
        );
        let tc = &completion["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_scan", "the first stream's call id survives");
        assert_eq!(
            tc["function"]["name"], "run_shell",
            "the first stream's call name survives"
        );
        assert_ne!(
            completion["choices"][0]["message"]["content"],
            serde_json::json!("second stream"),
            "the second stream's content must not leak into the result"
        );

        drop(tx);
        let mut retries = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Retrying { attempt, .. } = ev {
                retries.push(attempt);
            }
        }
        assert_eq!(retries, Vec::<u32>::new(), "no retry was signalled");
        server.abort();
    }

    /// Replay safety for a content turn: once the model has streamed answer
    /// tokens, those bytes are committed to the transcript. Resending would
    /// duplicate them, so a completion that produced content is returned
    /// verbatim and never re-requested. Proves committed tokens are never
    /// duplicated by the configurable retry loop.
    #[tokio::test]
    async fn a_turn_that_streamed_content_is_never_replayed() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        // The first request streams a real answer; any second request streams
        // different text. The turn must never be resent, so `served` stays at 1.
        let served = Arc::new(Mutex::new(0usize));
        let server_served = served.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut conn, _)) = listener.accept().await else {
                    break;
                };
                let served = server_served.clone();
                tokio::spawn(async move {
                    let mut scratch = [0u8; 4096];
                    let _ = conn.read(&mut scratch).await;
                    let n = {
                        let mut s = served.lock().await;
                        *s += 1;
                        *s
                    };
                    let sse = if n == 1 {
                        "data: {\"choices\":[{\"delta\":{\"content\":\"first answer\"},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n"
                    } else {
                        "data: {\"choices\":[{\"delta\":{\"content\":\"duplicated answer\"},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n"
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                        sse.len()
                    );
                    let _ = conn.write_all(resp.as_bytes()).await;
                    let _ = conn.flush().await;
                });
            }
        });

        let (tx, mut rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let completion = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(10, 0),
        )
        .await
        .expect("the first stream's content is returned, not replayed");

        assert_eq!(
            *served.lock().await, 1,
            "a completion that streamed content is never resent"
        );
        assert_eq!(
            completion["choices"][0]["message"]["content"], "first answer",
            "the returned content is the first stream's, not a duplicate"
        );

        drop(tx);
        let mut retries = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Retrying { attempt, .. } = ev {
                retries.push(attempt);
            }
        }
        assert_eq!(retries, Vec::<u32>::new(), "no retry was signalled");
        server.abort();
    }

    /// The scoping guarantee for #8781: the broad retry set is a CLI/TUI
    /// capability. A surface on [`RetryPolicy::legacy`] -- the desktop app and
    /// the OpenAI-compatible proxy -- must see a 429 and an answerless 200
    /// exactly as it did before this feature: a single error, no resend. Retry
    /// behaviour a user can neither configure nor watch is worse than the
    /// failure it hides, so it is not switched on for them silently.
    #[tokio::test]
    async fn the_legacy_policy_retries_neither_a_429_nor_a_blank_turn() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        // `first` decides what the very first request gets; every later request
        // streams a real answer. Under the legacy policy nothing should ever
        // reach that second response, and `served` proves it.
        async fn serve_once_then_answer(
            first: String,
        ) -> (String, Arc<Mutex<usize>>, tokio::task::JoinHandle<()>) {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let addr = listener.local_addr().expect("addr");
            let served = Arc::new(Mutex::new(0usize));
            let server_served = served.clone();
            let handle = tokio::spawn(async move {
                loop {
                    let Ok((mut conn, _)) = listener.accept().await else {
                        break;
                    };
                    let served = server_served.clone();
                    let first = first.clone();
                    tokio::spawn(async move {
                        let mut scratch = [0u8; 4096];
                        let _ = conn.read(&mut scratch).await;
                        let n = {
                            let mut s = served.lock().await;
                            *s += 1;
                            *s
                        };
                        if n == 1 {
                            let _ = conn.write_all(first.as_bytes()).await;
                        } else {
                            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"recovered\"},\"finish_reason\":\"stop\"}]}\n\n\
                                       data: [DONE]\n\n";
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                                sse.len()
                            );
                            let _ = conn.write_all(resp.as_bytes()).await;
                        }
                        let _ = conn.flush().await;
                    });
                }
            });
            (format!("http://{addr}/v1/chat/completions"), served, handle)
        }

        // A 429 that the configurable policy would ride out.
        let (url, served, server) = serve_once_then_answer(
            "HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\nRetry-After: 0\r\nContent-Length: 0\r\n\r\n"
                .to_string(),
        )
        .await;
        let (tx, mut rx) = sink();
        let err = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::legacy(),
        )
        .await
        .expect_err("legacy treats a 429 as fatal");
        assert!(err.contains("HTTP 429"), "names the upstream failure: {err}");
        assert_eq!(*served.lock().await, 1, "the request was not resent");
        drop(tx);
        assert!(
            !rx.try_recv()
                .is_ok_and(|ev| matches!(ev, StreamEvent::Retrying { .. })),
            "no attempt was announced, because none was made"
        );
        server.abort();

        // A blank 200: the very defect #8781 fixes, left in place on surfaces
        // that never opted into the fix.
        let blank = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                     data: [DONE]\n\n";
        let blank_resp = format!(
            "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{blank}",
            blank.len()
        );
        let (url, served, server) = serve_once_then_answer(blank_resp).await;
        let (tx, _rx) = sink();
        let completion = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::legacy(),
        )
        .await
        .expect("legacy returns the blank turn rather than retrying it");
        assert!(
            completion_is_answerless(&completion),
            "the blank turn came back as a success, unretried: {completion}"
        );
        assert_eq!(*served.lock().await, 1, "the blank turn was not resent");
        server.abort();
    }

    /// A dropped connection is the one cause the legacy policy does resend, and
    /// it gets exactly one attempt -- the pre-#8781 `send_with_one_retry`
    /// contract, still intact for the desktop and proxy surfaces.
    #[tokio::test]
    async fn the_legacy_policy_still_retries_one_dropped_connection() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.expect("first accept");
            let mut scratch = [0u8; 1024];
            let _ = first.read(&mut scratch).await;
            drop(first);

            let (mut second, _) = listener.accept().await.expect("second accept");
            let _ = second.read(&mut scratch).await;
            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                       data: [DONE]\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                sse.len()
            );
            let _ = second.write_all(resp.as_bytes()).await;
            let _ = second.flush().await;
        });

        let (tx, _rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let completion = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::legacy(),
        )
        .await
        .expect("one retry still carries a dropped connection");

        assert_eq!(completion["choices"][0]["message"]["content"], "hi");
        assert_eq!(
            RetryPolicy::legacy().max_retries,
            1,
            "one attempt after the first failure, as before #8781"
        );
        server.await.expect("server task");
    }

    #[test]
    fn completion_is_answerless_classifies_blank_turns() {
        // Tool calls with empty content are a real turn, not blank.
        let tool_turn = json!({
            "choices": [{
                "message": { "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": { "name": "read", "arguments": "{}" }
                }] },
                "finish_reason": "tool_calls"
            }]
        });
        assert!(!completion_is_answerless(&tool_turn));

        // A "length" finish is a real truncation, handled by the caller.
        let truncated = json!({
            "choices": [{ "message": { "content": null }, "finish_reason": "length" }]
        });
        assert!(!completion_is_answerless(&truncated));

        // A normal answer is not blank.
        let normal = json!({
            "choices": [{ "message": { "content": "hi" }, "finish_reason": "stop" }]
        });
        assert!(!completion_is_answerless(&normal));

        // Empty content with a "stop" finish IS the blank turn.
        let blank = json!({
            "choices": [{ "message": { "content": null }, "finish_reason": "stop" }]
        });
        assert!(completion_is_answerless(&blank));

        // An absent finish_reason is also blank.
        let no_finish = json!({
            "choices": [{ "message": { "content": null } }]
        });
        assert!(completion_is_answerless(&no_finish));

        // Whitespace-only content streamed bytes, so it is NOT blank: replaying
        // it would duplicate those tokens.
        let whitespace = json!({
            "choices": [{ "message": { "content": "   " }, "finish_reason": "stop" }]
        });
        assert!(!completion_is_answerless(&whitespace));

        // Content-part arrays reach this predicate through the shared loop
        // guard, where completions come from invokers other than the streaming
        // client. A populated array is a real answer; an empty one is blank.
        let parts = json!({
            "choices": [{
                "message": { "content": [{ "type": "text", "text": "hi" }] },
                "finish_reason": "stop"
            }]
        });
        assert!(!completion_is_answerless(&parts));
        let empty_parts = json!({
            "choices": [{ "message": { "content": [] }, "finish_reason": "stop" }]
        });
        assert!(completion_is_answerless(&empty_parts));
    }

    #[test]
    fn retry_max_zero_disables_retrying_entirely() {
        let policy = RetryPolicy::configurable(0, 0);
        assert!(!policy.allows(0), "no retries allowed when max_retries is 0");
    }

    /// `max_retries: 0` must fail after exactly one request, with no `Retrying`
    /// event and no sleep, even when the upstream keeps answering 429.
    #[tokio::test]
    async fn a_zero_retry_budget_answers_429_once_and_gives_up() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let count = Arc::new(Mutex::new(0usize));
        let server_count = count.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut conn, _)) = listener.accept().await else {
                    break;
                };
                *server_count.lock().await += 1;
                let mut scratch = [0u8; 4096];
                let _ = conn.read(&mut scratch).await;
                let _ = conn
                    .write_all(
                        b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\nRetry-After: 0\r\nContent-Length: 0\r\n\r\n",
                    )
                    .await;
                let _ = conn.flush().await;
            }
        });

        let (tx, mut rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let err = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(0, 0),
        )
        .await
        .expect_err("a disabled retry budget gives up on the first 429");

        assert!(err.contains("429"), "the upstream status survives: {err}");
        assert_eq!(*count.lock().await, 1, "exactly one request was sent");

        drop(tx);
        let mut retries = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Retrying { .. } = ev {
                retries.push(());
            }
        }
        assert!(retries.is_empty(), "no retry event with a zero budget");
        server.abort();
    }

    /// Exhaustion names the upstream failure verbatim: the always-429 server
    /// must surface as `HTTP 429` inside the exhaustion message.
    #[tokio::test]
    async fn retry_exhaustion_surfaces_the_upstream_failure() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::sync::Mutex;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let count = Arc::new(Mutex::new(0usize));
        let server_count = count.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut conn, _)) = listener.accept().await else {
                    break;
                };
                *server_count.lock().await += 1;
                let mut scratch = [0u8; 4096];
                let _ = conn.read(&mut scratch).await;
                let _ = conn
                    .write_all(
                        b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\nRetry-After: 0\r\nContent-Length: 0\r\n\r\n",
                    )
                    .await;
                let _ = conn.flush().await;
            }
        });

        let (tx, _rx) = sink();
        let url = format!("http://{addr}/v1/chat/completions");
        let err = stream_openai_chat_completions(
            &Client::new(),
            &url,
            &[],
            &json!({ "model": "m", "messages": [] }),
            &tx,
            RetryPolicy::configurable(2, 0),
        )
        .await
        .expect_err("a permanently-429 upstream exhausts the budget");

        assert!(err.contains("429"), "the upstream failure survives: {err}");
        assert_eq!(*count.lock().await, 3, "initial attempt plus two retries");
        server.abort();
    }

    #[test]
    fn retry_after_delay_parses_and_caps() {
        let mut headers = reqwest::header::HeaderMap::new();
        assert_eq!(retry_after_delay(&headers), None, "no header -> None");

        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("2"),
        );
        assert_eq!(
            retry_after_delay(&headers),
            Some(std::time::Duration::from_millis(2000))
        );

        // Garbage is not a delay.
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("soon"),
        );
        assert_eq!(retry_after_delay(&headers), None);

        // An HTTP-date is not accepted either.
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(retry_after_delay(&headers), None);

        // An absurd delay is capped.
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("86400"),
        );
        assert_eq!(
            retry_after_delay(&headers),
            Some(std::time::Duration::from_millis(60_000))
        );
    }

    #[test]
    fn retry_policy_delay_grows_caps_and_honours_retry_after() {
        let policy = RetryPolicy::configurable(10, 1000);
        // Grows with retries_used: attempt 0 -> ~1s band, attempt 3 -> ~8s band.
        let d0 = policy.delay(0, None).as_millis();
        let d3 = policy.delay(3, None).as_millis();
        assert!(d0 >= 500 && d0 <= 1000, "first backoff in [0.5x,1.0x]: {d0}");
        assert!(d3 >= 4000 && d3 <= 8000, "fourth backoff in [0.5x,1.0x]: {d3}");
        assert!(d3 > d0, "backoff grows with retries_used");

        // The 30s cap: the computed delay is capped before jittering, so the
        // result sits in the capped value's jitter band, never above the cap.
        let big = RetryPolicy::configurable(10, 100_000);
        let capped = big.delay(10, None).as_millis();
        assert!(
            (15_000..=30_000).contains(&capped),
            "capped backoff in [0.5x,1.0x] of 30s: {capped}"
        );

        // Retry-After overrides the computed value.
        let after = std::time::Duration::from_millis(1234);
        assert_eq!(policy.delay(5, Some(after)), after);

        // Zero backoff stays at zero (no jitter floor).
        let none = RetryPolicy::configurable(10, 0);
        assert_eq!(none.delay(7, None).as_millis(), 0);
    }

    /// A proxy in the environment breaks Jan and nothing else, and never shows up
    /// in the error. Names only: the values carry credentials.
    #[test]
    fn proxy_env_hint_names_set_variables_without_their_values() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("HTTPS_PROXY");
        std::env::remove_var("HTTPS_PROXY");
        let before = proxy_env_hint();

        std::env::set_var("HTTPS_PROXY", "http://user:secret@proxy.internal:8080");
        let hint = proxy_env_hint().expect("a set proxy is reported");
        assert!(hint.contains("HTTPS_PROXY"), "names the variable: {hint}");
        assert!(!hint.contains("secret"), "never prints the value: {hint}");

        std::env::set_var("HTTPS_PROXY", "   ");
        assert_eq!(proxy_env_hint(), before, "a blank value is not a proxy");

        match prev {
            Some(v) => std::env::set_var("HTTPS_PROXY", v),
            None => std::env::remove_var("HTTPS_PROXY"),
        }
    }

    /// A model served both by a Jan desktop API server (reachable over HTTP) and
    /// by local engine descriptors (no base_url) must always resolve to the
    /// server. `HashMap` order is randomized, so the local entries outnumber the
    /// remote one here: without a deterministic preference this fails most runs.
    #[cfg(feature = "cli")]
    #[tokio::test]
    async fn shared_model_id_resolves_to_the_reachable_provider() {
        let mut configs = HashMap::new();
        for local in ["llamacpp", "llamacpp-rs", "mlx", "engine-d", "engine-e"] {
            configs.insert(
                local.to_string(),
                ProviderConfig {
                    provider: local.into(),
                    base_url: None,
                    models: vec!["sentence-transformer-mini".into()],
                    ..Default::default()
                },
            );
        }
        configs.insert(
            "JanServer".to_string(),
            ProviderConfig {
                provider: "JanServer".into(),
                base_url: Some("http://127.0.0.1:1337/v1".into()),
                models: vec!["sentence-transformer-mini".into()],
                ..Default::default()
            },
        );

        let (url, _keys) = resolve_upstream_for_model(
            "sentence-transformer-mini",
            Arc::new(Mutex::new(configs)),
        )
        .await
        .expect("the API server provider is reachable");
        assert_eq!(url, "http://127.0.0.1:1337/v1/chat/completions");
    }

    #[test]
    fn parse_messages_passes_multimodal_content_array_through() {
        let messages = json!([{
            "role": "user",
            "content": [
                { "type": "text", "text": "look" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AA" } },
            ],
        }]);
        let out = parse_openai_messages(&messages).unwrap();
        assert_eq!(out[0]["content"], messages[0]["content"]);
    }

    #[test]
    fn parse_messages_rejects_missing_content() {
        let messages = json!([{ "role": "user" }]);
        assert!(parse_openai_messages(&messages).is_err());
    }

    #[test]
    fn parse_messages_allows_null_content_assistant_tool_call_turn() {
        // An assistant tool-call turn (content: null + tool_calls) round-trips
        // through history and must be re-parseable on a follow-up request.
        let messages = json!([{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "write", "arguments": "{}" }
            }]
        }]);
        let out = parse_openai_messages(&messages).unwrap();
        assert_eq!(out[0]["role"], "assistant");
        assert!(out[0]["content"].is_null());
        assert_eq!(out[0]["tool_calls"][0]["id"], "call_1");
    }

    #[test]
    fn parse_messages_preserves_tool_result_message() {
        // A role:tool result must keep its tool_call_id so the assistant/tool
        // pairing stays valid when the conversation is re-submitted.
        let messages = json!([{
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "wrote file"
        }]);
        let out = parse_openai_messages(&messages).unwrap();
        assert_eq!(out[0]["role"], "tool");
        assert_eq!(out[0]["tool_call_id"], "call_1");
        assert_eq!(out[0]["content"], "wrote file");
    }

    #[test]
    fn parse_messages_still_rejects_null_content_without_tool_calls() {
        // A plain assistant/user turn with null content is still invalid.
        let messages = json!([{ "role": "assistant", "content": null }]);
        assert!(parse_openai_messages(&messages).is_err());
    }

    #[test]
    fn repair_leaves_well_formed_conversation_untouched() {
        let mut messages = vec![
            json!({ "role": "user", "content": "hi" }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "read", "arguments": "{}" }
                }]
            }),
            json!({ "role": "tool", "tool_call_id": "call_1", "content": "file contents" }),
            json!({ "role": "assistant", "content": "done" }),
        ];
        let before = messages.clone();
        assert_eq!(repair_dangling_tool_calls(&mut messages), 0);
        assert_eq!(messages, before);
    }

    #[test]
    fn repair_inserts_synthetic_result_for_a_fully_missing_tool_reply() {
        // An `ask`-style call interrupted before it ever produced a result --
        // e.g. the process was killed while the prompt was still pending.
        let mut messages = vec![
            json!({ "role": "user", "content": "make cat slide" }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "toolu_ask",
                    "type": "function",
                    "function": { "name": "ask", "arguments": "{}" }
                }]
            }),
            json!({ "role": "user", "content": "next message, no reply ever recorded" }),
        ];
        assert_eq!(repair_dangling_tool_calls(&mut messages), 1);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "toolu_ask");
        assert!(messages[2]["content"].as_str().unwrap().starts_with("ERROR"));
        assert_eq!(messages[3]["content"], "next message, no reply ever recorded");
    }

    #[test]
    fn repair_fills_only_the_missing_id_in_a_multi_call_turn() {
        // Two tool calls in one turn; only one got a reply before the gap.
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    { "id": "call_a", "type": "function", "function": { "name": "read", "arguments": "{}" } },
                    { "id": "call_b", "type": "function", "function": { "name": "ask", "arguments": "{}" } },
                ]
            }),
            json!({ "role": "tool", "tool_call_id": "call_a", "content": "file contents" }),
        ];
        assert_eq!(repair_dangling_tool_calls(&mut messages), 1);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["tool_call_id"], "call_a");
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_b");
    }

    #[test]
    fn repair_handles_a_dangling_call_at_the_end_of_the_conversation() {
        // No trailing message at all after the unanswered tool call.
        let mut messages = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "ask", "arguments": "{}" }
            }]
        })];
        assert_eq!(repair_dangling_tool_calls(&mut messages), 1);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_1");
    }

    #[test]
    fn sanitize_leaves_valid_tool_calls_untouched() {
        // Well-formed arguments, plus the empty-object and absent-arguments
        // spellings providers use for a no-argument call.
        let mut messages = vec![
            json!({ "role": "user", "content": "hi" }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    { "id": "c1", "type": "function", "function": { "name": "read", "arguments": "{\"path\":\"a.rs\"}" } },
                    { "id": "c2", "type": "function", "function": { "name": "ls", "arguments": "{}" } },
                    { "id": "c3", "type": "function", "function": { "name": "now" } },
                ]
            }),
            json!({ "role": "tool", "tool_call_id": "c1", "content": "ok" }),
        ];
        let before = messages.clone();
        assert_eq!(drop_malformed_tool_calls(&mut messages), 0);
        assert_eq!(messages, before);
    }

    #[test]
    fn sanitize_drops_a_truncated_call_and_its_result() {
        // The wedging case: a stream cut mid-argument, persisted, then resent
        // on every later turn and 422'd by the upstream.
        let mut messages = vec![
            json!({ "role": "user", "content": "write the file" }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_bad",
                    "type": "function",
                    "function": { "name": "write", "arguments": "{\"path\":\"a.rs\",\"content\":\"fn ma" }
                }]
            }),
            json!({ "role": "tool", "tool_call_id": "call_bad", "content": "stale" }),
            json!({ "role": "user", "content": "still there?" }),
        ];
        assert_eq!(drop_malformed_tool_calls(&mut messages), 1);
        // The empty assistant shell and the orphaned result are both gone.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"], "write the file");
        assert_eq!(messages[1]["content"], "still there?");
    }

    #[test]
    fn sanitize_keeps_valid_siblings_of_a_poisoned_call() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    { "id": "ok", "type": "function", "function": { "name": "read", "arguments": "{\"path\":\"a\"}" } },
                    { "id": "bad", "type": "function", "function": { "name": "write", "arguments": "{\"path\":\"b" } },
                ]
            }),
            json!({ "role": "tool", "tool_call_id": "ok", "content": "contents" }),
            json!({ "role": "tool", "tool_call_id": "bad", "content": "stale" }),
        ];
        assert_eq!(drop_malformed_tool_calls(&mut messages), 1);
        let calls = messages[0]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "ok");
        // Only the poisoned call's result was dropped.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["tool_call_id"], "ok");
    }

    #[test]
    fn sanitize_preserves_assistant_text_when_all_calls_are_dropped() {
        let mut messages = vec![json!({
            "role": "assistant",
            "content": "I'll write that file now.",
            "tool_calls": [{
                "id": "bad",
                "type": "function",
                "function": { "name": "write", "arguments": "{\"content\":\"trunc" }
            }]
        })];
        assert_eq!(drop_malformed_tool_calls(&mut messages), 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "I'll write that file now.");
        assert!(messages[0].get("tool_calls").is_none());
    }

    #[test]
    fn sanitize_then_dangling_repair_leaves_a_sendable_conversation() {
        // The two passes compose the way the orchestrator runs them: the
        // poisoned call goes away, and the surviving call that lost its result
        // gets the synthetic error reply rather than being left dangling.
        let mut messages = vec![
            json!({ "role": "user", "content": "go" }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    { "id": "keep", "type": "function", "function": { "name": "read", "arguments": "{}" } },
                    { "id": "poison", "type": "function", "function": { "name": "write", "arguments": "{\"a\":" } },
                ]
            }),
        ];
        assert_eq!(drop_malformed_tool_calls(&mut messages), 1);
        assert_eq!(repair_dangling_tool_calls(&mut messages), 1);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "keep");
        // Every remaining call id has exactly one matching result.
        let ids: Vec<&str> = messages[1]["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["keep"]);
    }

    #[test]
    fn detects_provider_context_overflow_bodies() {
        assert!(is_context_overflow_body(
            "{\"error\":{\"code\":\"context_length_exceeded\"}}"
        ));
        assert!(is_context_overflow_body("This model's maximum context length is 8192 tokens"));
        assert!(is_context_overflow_body("prompt is too long: 210000 tokens > 200000"));
        assert!(is_context_overflow_body(
            "the request exceeds the available context size"
        ));
        assert!(is_context_overflow_body(
            "{\"error\":{\"code\":400,\"message\":\"request (4267 tokens) exceeds the available \
             context size (4096 tokens), try increasing it\",\"type\":\"exceed_context_size_error\",\
             \"n_prompt_tokens\":4267,\"n_ctx\":4096}}"
        ));
        assert!(!is_context_overflow_body("invalid api key"));
        assert!(!is_context_overflow_body("rate limit exceeded"));
    }

    #[test]
    fn overflow_marker_round_trips() {
        let err = format!("[{CONTEXT_OVERFLOW_MARKER}] Upstream returned HTTP 400: ...");
        assert!(is_context_overflow_error(&err));
        assert!(!is_context_overflow_error("Upstream returned HTTP 500: boom"));
    }

    /// The shapes strict endpoints actually return, plus the two ways a false
    /// positive would arise: an error that names the field without rejecting it,
    /// and a rejection of some other field.
    #[test]
    fn detects_a_rejected_reasoning_content_field() {
        for body in [
            "Upstream returned HTTP 400: {\"error\":{\"message\":\"'messages.1' : for 'role':'assistant' the following must be satisfied[('messages.1.reasoning_content' : property 'reasoning_content' is unsupported)]\"}}",
            "Upstream returned HTTP 400: Unrecognized request argument supplied: reasoning_content",
            "Upstream returned HTTP 400: body.messages.1.reasoning_content: Extra inputs are not permitted",
            "Upstream returned HTTP 400: Invalid value for 'reasoning_content'",
        ] {
            assert!(is_reasoning_field_error(body), "missed: {body}");
        }
        assert!(!is_reasoning_field_error(
            "Upstream returned HTTP 500: reasoning_content was truncated"
        ));
        assert!(!is_reasoning_field_error(
            "Upstream returned HTTP 400: property 'audio' is unsupported"
        ));
    }

    #[test]
    fn accumulates_content_and_emits_token_per_delta() {
        let (tx, mut rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "content": "Hel" } }] }).to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": { "content": "lo" } }] }).to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": {}, "finish_reason": "stop" }] }).to_string(),
            &tx,
        );

        let completion = acc.into_completion();
        assert_eq!(completion["choices"][0]["message"]["content"], "Hello");
        assert_eq!(completion["choices"][0]["finish_reason"], "stop");
        assert!(completion["choices"][0]["message"]
            .get("tool_calls")
            .is_none());

        drop(tx);
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Token { text } = ev {
                tokens.push(text);
            }
        }
        assert_eq!(tokens, vec!["Hel", "lo"]);
    }

    #[test]
    fn reasoning_content_streams_as_native_reasoning_events_and_stays_out_of_content() {
        let (tx, mut rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "reasoning_content": "let me " } }] }).to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": { "reasoning_content": "think" } }] }).to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": { "content": "answer" } }] }).to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": {}, "finish_reason": "stop" }] }).to_string(),
            &tx,
        );

        // Reasoning stays out of the answer prose, but is preserved on the
        // reconstructed message so a caller can resend it as history.
        let completion = acc.into_completion();
        assert_eq!(completion["choices"][0]["message"]["content"], "answer");
        assert_eq!(
            completion["choices"][0]["message"]["reasoning_content"],
            "let me think"
        );

        drop(tx);
        let mut reasoning = Vec::new();
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Reasoning { text } => reasoning.push(text),
                StreamEvent::Token { text } => tokens.push(text),
                _ => {}
            }
        }
        // Native events: no synthetic <think> wrapper tokens on the content
        // stream, so a consumer never has to re-parse tags it did not receive.
        assert_eq!(reasoning, vec!["let me ", "think"]);
        assert_eq!(tokens, vec!["answer"]);
    }

    #[test]
    fn reasoning_only_completion_emits_reasoning_and_no_tokens() {
        let (tx, mut rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "reasoning_content": "hmm" } }] }).to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": {}, "finish_reason": "stop" }] }).to_string(),
            &tx,
        );

        let completion = acc.into_completion();
        assert!(completion["choices"][0]["message"]["content"].is_null());

        drop(tx);
        let mut reasoning = Vec::new();
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Reasoning { text } => reasoning.push(text),
                StreamEvent::Token { text } => tokens.push(text),
                _ => {}
            }
        }
        assert_eq!(reasoning, vec!["hmm"]);
        assert!(tokens.is_empty(), "no content tokens for a reasoning-only turn: {tokens:?}");
    }

    /// A completion with no `reasoning_content` deltas must keep exactly the
    /// shape it always had: the field is omitted, not emitted empty, so a
    /// provider that rejects unknown assistant keys is unaffected.
    #[test]
    fn a_turn_without_reasoning_omits_the_field_entirely() {
        let (tx, _rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "content": "answer" } }] }).to_string(),
            &tx,
        );
        let completion = acc.into_completion();
        let message = &completion["choices"][0]["message"];
        assert_eq!(message["content"], "answer");
        assert!(
            message.get("reasoning_content").is_none(),
            "no reasoning streamed, so the key must be absent: {message}"
        );
    }

    /// A tool-call turn's reasoning is preserved too: the model reasons about
    /// which tool to call, and that reasoning has to survive onto the assistant
    /// turn the loop resends with its `tool_calls`.
    #[test]
    fn reasoning_is_preserved_on_a_tool_call_turn() {
        let (tx, _rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "reasoning_content": "need to grep" } }] })
                .to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": { "tool_calls": [{
                "index": 0,
                "id": "c1",
                "function": { "name": "bash", "arguments": "{}" }
            }] } }] })
            .to_string(),
            &tx,
        );
        let completion = acc.into_completion();
        let message = &completion["choices"][0]["message"];
        assert_eq!(message["reasoning_content"], "need to grep");
        assert_eq!(message["tool_calls"][0]["id"], "c1");
    }

    /// A resent assistant turn keeps its `reasoning_content` through the message
    /// normalizer. Local llama.cpp templates with `preserve_thinking` re-emit
    /// prior reasoning from this field; dropping it would shrink earlier turns
    /// and force the KV-cache prefix to be reprocessed.
    #[test]
    fn parse_messages_preserves_assistant_reasoning_content() {
        let messages = json!([{
            "role": "assistant",
            "content": "the answer",
            "reasoning_content": "the thinking"
        }]);
        let out = parse_openai_messages(&messages).unwrap();
        assert_eq!(out[0]["content"], "the answer");
        assert_eq!(out[0]["reasoning_content"], "the thinking");
    }

    /// Only assistant turns carry reasoning back. A stray field on another role
    /// is not part of the protocol, so it is dropped rather than forwarded.
    #[test]
    fn parse_messages_drops_reasoning_on_non_assistant_roles() {
        let messages = json!([
            { "role": "user", "content": "q", "reasoning_content": "nope" },
            { "role": "assistant", "content": "a" },
        ]);
        let out = parse_openai_messages(&messages).unwrap();
        assert!(out[0].get("reasoning_content").is_none());
        assert!(
            out[1].get("reasoning_content").is_none(),
            "an assistant turn with no reasoning stays unchanged"
        );
    }

    /// Providers that inline `<think>` tags in `content` (no reasoning_content
    /// field) keep streaming through Token untouched: the tag-stripping
    /// fallback lives in the consumers.
    #[test]
    fn inline_think_tags_in_content_pass_through_as_tokens() {
        let (tx, mut rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "content": "<think>hmm</think>answer" } }] })
                .to_string(),
            &tx,
        );
        drop(tx);
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Token { text } => tokens.push(text),
                StreamEvent::Reasoning { .. } => panic!("no native reasoning here"),
                _ => {}
            }
        }
        assert_eq!(tokens, vec!["<think>hmm</think>answer"]);
    }

    #[test]
    fn reassembles_tool_call_arguments_split_across_deltas() {
        let (tx, _rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "search", "arguments": "{\"q\":" } }
            ] } }] })
            .to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "\"rust\"}" } }
            ] } }] })
            .to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }],
                     "usage": { "total_tokens": 12 } })
            .to_string(),
            &tx,
        );

        let completion = acc.into_completion();
        let tc = &completion["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "search");
        assert_eq!(tc["function"]["arguments"], "{\"q\":\"rust\"}");
        assert_eq!(completion["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(completion["usage"]["total_tokens"], 12);
    }

    #[test]
    fn emits_tool_call_started_once_when_id_and_name_first_known() {
        let (tx, mut rx) = sink();
        let mut acc = SseAccumulator::default();
        // First delta carries id + name; arguments arrive later, split across
        // deltas -- the in-progress signal must fire on this first delta only.
        acc.ingest(
            &json!({ "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "write", "arguments": "{\"path\":" } }
            ] } }] })
            .to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "\"a.txt\"}" } }
            ] } }] })
            .to_string(),
            &tx,
        );

        drop(tx);
        let started: Vec<(String, String)> = std::iter::from_fn(|| rx.try_recv().ok())
            .filter_map(|ev| match ev {
                StreamEvent::ToolCallStarted { id, name } => Some((id, name)),
                _ => None,
            })
            .collect();
        assert_eq!(started, vec![("call_1".to_string(), "write".to_string())]);
    }

    #[test]
    fn emits_tool_call_started_per_parallel_call() {
        let (tx, mut rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call_a", "function": { "name": "read", "arguments": "" } },
                { "index": 1, "id": "call_b", "function": { "name": "grep", "arguments": "" } }
            ] } }] })
            .to_string(),
            &tx,
        );

        drop(tx);
        let mut names: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
            .filter_map(|ev| match ev {
                StreamEvent::ToolCallStarted { name, .. } => Some(name),
                _ => None,
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["grep".to_string(), "read".to_string()]);
    }

    #[test]
    fn captures_mid_stream_error_object_with_type_prefix() {
        let (tx, _rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "choices": [{ "delta": { "content": "partial" } }] }).to_string(),
            &tx,
        );
        acc.ingest(
            &json!({ "error": { "message": "upstream exploded", "type": "server_error" } })
                .to_string(),
            &tx,
        );
        assert_eq!(
            acc.error.as_deref(),
            Some("server_error: upstream exploded")
        );
    }

    #[test]
    fn mid_stream_error_falls_back_to_message_without_type() {
        let (tx, _rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest(
            &json!({ "error": { "message": "boom" } }).to_string(),
            &tx,
        );
        assert_eq!(acc.error.as_deref(), Some("boom"));
    }

    #[test]
    fn ignores_done_sentinel_and_malformed_lines() {
        let (tx, _rx) = sink();
        let mut acc = SseAccumulator::default();
        acc.ingest("[DONE]", &tx);
        acc.ingest("not json", &tx);
        let completion = acc.into_completion();
        assert!(completion["choices"][0]["message"]["content"].is_null());
    }

    /// Feed arbitrary byte chunks through the real buffering path, then close.
    fn feed_and_close(
        chunks: &[&str],
        events: &mpsc::UnboundedSender<StreamEvent>,
    ) -> serde_json::Value {
        let mut buf = String::new();
        let mut acc = SseAccumulator::default();
        for c in chunks {
            buf.push_str(c);
            drain_complete_lines(&mut buf, &mut acc, events);
        }
        flush_trailing_line(&buf, &mut acc, events);
        acc.into_completion()
    }

    #[test]
    fn flushes_final_line_without_trailing_newline() {
        let (tx, _rx) = sink();
        // Provider closes right after the final data line: no `\n`, no `[DONE]`.
        let final_chunk = json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] });
        let completion = feed_and_close(
            &[
                "data: ",
                &json!({ "choices": [{ "delta": { "content": "hi" } }] }).to_string(),
                "\n\ndata: ",
                &final_chunk.to_string(),
            ],
            &tx,
        );

        assert_eq!(completion["choices"][0]["message"]["content"], "hi");
        assert_eq!(completion["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn newline_terminated_stream_still_parses() {
        let (tx, _rx) = sink();
        let completion = feed_and_close(
            &[
                &format!(
                    "data: {}\n\n",
                    json!({ "choices": [{ "delta": { "content": "ok" }, "finish_reason": "stop" }] })
                ),
                "data: [DONE]\n\n",
            ],
            &tx,
        );
        assert_eq!(completion["choices"][0]["message"]["content"], "ok");
        assert_eq!(completion["choices"][0]["finish_reason"], "stop");
    }
}
