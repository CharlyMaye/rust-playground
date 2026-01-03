use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct IndexResponse {
     pub user_id: Option<String>,
    pub session_counter: i32,
}