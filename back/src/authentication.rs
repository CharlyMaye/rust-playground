use actix_session::{Session, SessionMiddleware, storage::RedisSessionStore};
use actix_web::{
    App, HttpResponse, HttpServer, Result, middleware, web,
    web::{get, post, resource},
};
use serde::{Deserialize};

use crate::shared::IndexResponse;

#[derive(Deserialize)]
struct Identity {
    user_id: String,
}

async fn login(user_id: web::Json<Identity>, session: Session) -> Result<HttpResponse> {
    let id = user_id.into_inner().user_id;
    session.insert("user_id", &id)?;
    session.renew();

    let counter: i32 = session
        .get::<i32>("counter")
        .unwrap_or(Some(0))
        .unwrap_or(0);

    Ok(HttpResponse::Ok().json(IndexResponse {
        user_id: Some(id),
        session_counter: counter,
    }))
}

async fn logout(session: Session) -> Result<String> {
    let id: Option<String> = session.get("user_id")?;
    if let Some(x) = id {
        session.purge();
        Ok(format!("Logged out: {x}"))
    } else {
        Ok("Could not log out anonymous user".into())
    }
}

pub fn authentication_config(cfg: &mut web::ServiceConfig) {
    cfg.route("/login", web::post().to(login));
    cfg.route("/logout", web::post().to(logout));
}
