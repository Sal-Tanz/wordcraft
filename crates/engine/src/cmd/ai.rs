//! AI Assistant agent: full control assistant that interacts with an OpenAI-compatible API
//! endpoint and executes engine commands or text manipulations.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{CmdError, CmdResult, CommandSpec, Session, p};

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec::new("ai.prompt", "AI Prompt", "AI Assistant", ai_prompt)
            .params(r#"{"prompt": string, "apiUrl"?: string, "apiKey"?: string, "model"?: string, "systemPrompt"?: string, "apply"?: bool} — sends prompt to OpenAI-compatible API to generate commands or text. If apply is true (default true), executes generated engine commands."#),
        CommandSpec::new("ai.settings", "AI Settings", "AI Assistant", ai_settings)
            .params(r#"{"apiUrl"?: string, "apiKey"?: string, "model"?: string} — sets or reads AI Assistant configuration."#),
    ]
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

/// Configure or inspect AI Assistant settings.
fn ai_settings(s: &mut Session, v: &Value) -> CmdResult {
    let mut changed = false;
    if let Some(url) = v.get("apiUrl") {
        if url.is_null() || (url.is_string() && url.as_str().unwrap_or("").trim().is_empty()) {
            s.prefs.ai_api_url = None;
        } else if let Some(url_str) = url.as_str() {
            s.prefs.ai_api_url = Some(url_str.trim().to_string());
        } else {
            return Err(CmdError::Params("`apiUrl` must be a string".into()));
        }
        changed = true;
    }
    if let Some(key) = v.get("apiKey") {
        if key.is_null() || (key.is_string() && key.as_str().unwrap_or("").trim().is_empty()) {
            s.prefs.ai_api_key = None;
        } else if let Some(key_str) = key.as_str() {
            s.prefs.ai_api_key = Some(key_str.trim().to_string());
        } else {
            return Err(CmdError::Params("`apiKey` must be a string".into()));
        }
        changed = true;
    }
    if let Some(m) = v.get("model") {
        if m.is_null() || (m.is_string() && m.as_str().unwrap_or("").trim().is_empty()) {
            s.prefs.ai_model = None;
        } else if let Some(m_str) = m.as_str() {
            s.prefs.ai_model = Some(m_str.trim().to_string());
        } else {
            return Err(CmdError::Params("`model` must be a string".into()));
        }
        changed = true;
    }

    Ok(json!({
        "changed": changed,
        "apiUrl": s.prefs.ai_api_url,
        "apiKey": s.prefs.ai_api_key,
        "model": s.prefs.ai_model,
    }))
}

/// Process an AI Assistant prompt and optionally apply full control document actions.
fn ai_prompt(s: &mut Session, v: &Value) -> CmdResult {
    let prompt = p::req_str(v, "prompt")?.trim();
    if prompt.is_empty() {
        return Err(CmdError::Params("`prompt` cannot be empty".into()));
    }

    let api_url = p::str(v, "apiUrl")
        .map(|s| s.to_string())
        .or_else(|| s.prefs.ai_api_url.clone())
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

    let api_key = p::str(v, "apiKey")
        .map(|s| s.to_string())
        .or_else(|| s.prefs.ai_api_key.clone())
        .ok_or_else(|| CmdError::Failed("AI API Key is required. Please set `apiKey` or configure via `ai.settings` / `file.options`.".into()))?;

    let model = p::str(v, "model")
        .map(|s| s.to_string())
        .or_else(|| s.prefs.ai_model.clone())
        .unwrap_or_else(|| "gpt-4o".to_string());

    let apply = p::bool(v, "apply").unwrap_or(true);

    let doc_plain_text = s.doc.plain_text(wordcraft_doc::StoryRef::Body);
    let default_system_prompt = format!(
        "You are an AI assistant for WordCraft word processor with full control over the document.\n\
        The current document body plain text is:\n\"\"\"\n{}\n\"\"\"\n\n\
        When asked to edit, format, or generate content, respond with JSON containing:\n\
        1. \"explanation\": a brief summary of what was done or answered.\n\
        2. \"commands\": a list of engine commands to execute to apply the changes.\n\
        Example command formats in JSON:\n\
        - {{\"command\": \"text.insert\", \"params\": {{\"text\": \"Hello world\"}}}}\n\
        - {{\"command\": \"format.bold\", \"params\": {{}}}}\n\
        - {{\"command\": \"select.all\", \"params\": {{}}}}\n\
        - {{\"command\": \"para.align\", \"params\": {{\"align\": \"center\"}}}}\n\
        - {{\"command\": \"edit.replaceAll\", \"params\": {{\"text\": \"old\", \"with\": \"new\"}}}}\n\
        Return strictly valid JSON in the format: {{\"explanation\": \"...\", \"commands\": [...]}}",
        doc_plain_text
    );

    let system_prompt = p::str(v, "systemPrompt")
        .map(|s| s.to_string())
        .unwrap_or(default_system_prompt);

    let endpoint = if api_url.ends_with('/') {
        format!("{}chat/completions", api_url)
    } else if api_url.ends_with("/chat/completions") {
        api_url.clone()
    } else {
        format!("{}/chat/completions", api_url)
    };

    let req_body = ChatCompletionRequest {
        model: model.clone(),
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: system_prompt,
            },
            ChatMessage {
                role: "user".into(),
                content: prompt.to_string(),
            },
        ],
        temperature: Some(0.2),
    };

    let mut resp = ureq::post(&endpoint)
        .header("Authorization", &format!("Bearer {}", api_key))
        .send_json(&req_body)
        .map_err(|e| CmdError::Failed(format!("API request failed to {endpoint}: {e}")))?;

    let parsed_resp: ChatCompletionResponse = resp.body_mut().read_json()
        .map_err(|e| CmdError::Failed(format!("failed to parse API JSON response: {e}")))?;

    let content_str = parsed_resp.choices.first()
        .and_then(|c| c.message.content.as_deref())
        .ok_or_else(|| CmdError::Failed("empty content returned from AI API".into()))?;

    // Attempt to parse content as JSON response with commands
    let clean_json_str = if let Some(start) = content_str.find('{') {
        if let Some(end) = content_str.rfind('}') {
            &content_str[start..=end]
        } else {
            content_str
        }
    } else {
        content_str
    };

    let mut executed_results = Vec::new();
    let mut explanation = content_str.to_string();

    if let Ok(parsed_val) = serde_json::from_str::<Value>(clean_json_str) {
        if let Some(exp) = parsed_val.get("explanation").and_then(|e| e.as_str()) {
            explanation = exp.to_string();
        }

        if apply {
            if let Some(cmds) = parsed_val.get("commands").and_then(|c| c.as_array()) {
                for cmd_val in cmds {
                    if let Some(cmd_name) = cmd_val.get("command").and_then(|c| c.as_str()) {
                        let empty_params = json!({});
                        let params = cmd_val.get("params").unwrap_or(&empty_params);
                        match s.run(cmd_name, params) {
                            Ok(res) => executed_results.push(json!({"command": cmd_name, "status": "ok", "result": res})),
                            Err(err) => executed_results.push(json!({"command": cmd_name, "status": "error", "error": err.to_string()})),
                        }
                    }
                }
            }
        }
    } else if apply {
        // If non-JSON output, fallback to inserting the response text at current selection
        match s.run("text.insert", &json!({"text": content_str})) {
            Ok(res) => executed_results.push(json!({"command": "text.insert", "status": "ok", "result": res})),
            Err(err) => executed_results.push(json!({"command": "text.insert", "status": "error", "error": err.to_string()})),
        }
    }

    Ok(json!({
        "rawResponse": content_str,
        "explanation": explanation,
        "applied": apply,
        "executed": executed_results,
    }))
}
