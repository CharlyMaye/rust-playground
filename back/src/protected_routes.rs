// Exemples d'utilisation de l'authentification

use actix_web::{HttpResponse, Responder};
use crate::authentication::AuthenticatedUser;

// ============= EXTRACTOR =============
// Route protégée avec extractor : vérification individuelle
// Usage: .route("/protected", web::get().to(protected_route))
pub async fn protected_route(user: AuthenticatedUser) -> impl Responder {
    // user.user_id est garanti d'exister, session déjà renouvelée automatiquement
    HttpResponse::Ok().json(serde_json::json!({
        "message": "Route protégée par extractor",
        "user_id": user.user_id
    }))
}

// ============= MIDDLEWARE =============
// Route dans un scope protégé par middleware
// Usage: web::scope("/api").wrap(AuthenticationMiddleware).route("/data", web::get().to(api_data))
pub async fn api_data() -> impl Responder {
    // Pas besoin de vérifier la session, le middleware l'a déjà fait
    // On ne peut pas accéder directement au user_id ici sans l'extractor
    HttpResponse::Ok().json(serde_json::json!({
        "message": "Route protégée par middleware sur le scope",
        "data": "Données sécurisées"
    }))
}

// On peut combiner les deux pour avoir l'user_id dans une route protégée par middleware
pub async fn api_user_info(user: AuthenticatedUser) -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "message": "Route avec middleware + extractor",
        "user_id": user.user_id,
        "info": "Double protection"
    }))
}
