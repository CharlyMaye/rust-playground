use actix_session::{Session};
use actix_web::{
    HttpResponse, Result, web,
    FromRequest, HttpRequest, Error,
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
};
use std::future::{ready, Ready};
use serde::{Deserialize};

use crate::shared::IndexResponse;

// ============= FONCTION PARTAGÉE =============
// Logique commune de vérification et renouvellement de session
fn verify_and_renew_session(session: &Session) -> Result<String, Error> {
    match session.get::<String>("user_id") {
        Ok(Some(user_id)) => {
            session.renew();
            Ok(user_id)
        }
        _ => Err(actix_web::error::ErrorUnauthorized("Not authenticated"))
    }
}

// ============= EXTRACTOR =============
// Extractor pour extraire l'utilisateur authentifié de la session
pub struct AuthenticatedUser {
    pub user_id: String,
}

impl FromRequest for AuthenticatedUser {
    type Error = Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut actix_web::dev::Payload) -> Self::Future {
        let session = Session::extract(req).into_inner().unwrap();
        
        match verify_and_renew_session(&session) {
            Ok(user_id) => ready(Ok(AuthenticatedUser { user_id })),
            Err(e) => ready(Err(e))
        }
    }
}

// ============= MIDDLEWARE =============
// Middleware pour protéger un scope entier
pub struct AuthenticationMiddleware;

impl<S, B> Transform<S, ServiceRequest> for AuthenticationMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthenticationMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthenticationMiddlewareService { service }))
    }
}

pub struct AuthenticationMiddlewareService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for AuthenticationMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>>>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let session = Session::extract(req.request()).into_inner().unwrap();
        
        match verify_and_renew_session(&session) {
            Ok(_) => {
                let fut = self.service.call(req);
                Box::pin(async move {
                    let res = fut.await?;
                    Ok(res)
                })
            }
            Err(e) => {
                Box::pin(async move {
                    Err(e)
                })
            }
        }
    }
}

// ============= ROUTES DE LOGIN/LOGOUT =============
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
        counter,
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

// Configuration des routes d'authentification
pub fn authentication_config(cfg: &mut web::ServiceConfig) {
    cfg.route("/login", web::post().to(login));
    cfg.route("/logout", web::post().to(logout));
}
