mod docker;
mod docker_compose;

use docker_compose::{DockerComposeManager, Service};
use std::collections::HashMap;

use docker::{DockerManager, DockerImageManager, DockerContainerManager, DockerNetworkManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Décommenter pour tester Docker Compose
    test_docker_compose().await;

    // Décommenter pour tester Docker direct
    // test_docker().await;

    Ok(())
}

async fn test_docker_compose() {
    // Chemin vers le docker-compose.yml existant
    let compose_manager = DockerComposeManager::new("../docker-compose.yml");

    // Lister les services existants
    println!("\n📋 Services actuels:");
    compose_manager.list_services().unwrap();

    // Ajouter un nouveau service PostgreSQL
    println!("\n➕ Ajout d'un service PostgreSQL...");
    let mut env = HashMap::new();
    env.insert("POSTGRES_PASSWORD".to_string(), "secret".to_string());
    env.insert("POSTGRES_USER".to_string(), "admin".to_string());
    env.insert("POSTGRES_DB".to_string(), "mydb".to_string());

    let postgres_service = Service {
        image: Some("postgres:15-alpine".to_string()),
        build: None,
        container_name: Some("my-postgres".to_string()),
        ports: Some(vec!["5432:5432".to_string()]),
        environment: Some(env),
        networks: None,
        volumes: Some(vec!["postgres_data:/var/lib/postgresql/data".to_string()]),
        depends_on: None,
        command: None,
    };

    compose_manager.add_service("postgres", postgres_service).unwrap();

    // Ajouter le volume pour postgres
    println!("\n➕ Ajout du volume postgres_data...");
    compose_manager.add_volume("postgres_data").unwrap();

    // Lister les services après ajout
    println!("\n📋 Services après ajout:");
    compose_manager.list_services().unwrap();

    // Démarrer tous les services
    println!("\n🚀 Démarrage de tous les services...");
    compose_manager.up(true).unwrap();

    // Attendre un peu
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // Redémarrer un service spécifique
    println!("\n🔄 Redémarrage du service redis...");
    compose_manager.restart_service("redis").unwrap();

    // Arrêter un service spécifique
    println!("\n⏹️  Arrêt du service postgres...");
    compose_manager.stop_service("postgres").unwrap();

    // Supprimer le service postgres
    println!("\n➖ Suppression du service postgres...");
    compose_manager.remove_service("postgres").unwrap();

    // Lister les services après suppression
    println!("\n📋 Services après suppression:");
    compose_manager.list_services().unwrap();

    // Arrêter tous les services
    println!("\n⏹️  Arrêt de tous les services...");
    compose_manager.down().unwrap();
}

async fn test_docker() {
    let sleep_time = std::time::Duration::from_secs(1);
    let image_name = "hello-world";
    let network_name = "my_network"; 

    let docker_manager = DockerManager::new();
    docker_manager.docker_info().await;

    docker_manager.list_images().await;
    docker_manager.create_image(image_name).await;

    docker_manager.create_network(network_name).await;
    docker_manager.list_networks().await;
    
    let container_id = docker_manager.create_container(image_name, "my_container", None).await;
    let container_id_with_network = docker_manager.create_container(image_name, "my_container_with_network", Some(network_name)).await;
    tokio::time::sleep(sleep_time).await;
    docker_manager.list_containers().await;


    docker_manager.start_container(&container_id).await;
    tokio::time::sleep(sleep_time).await;

    docker_manager.stop_container(&container_id).await;
    tokio::time::sleep(sleep_time).await;

    docker_manager.remove_container(&container_id).await;
    docker_manager.remove_container(&container_id_with_network).await;
    tokio::time::sleep(sleep_time).await;

    docker_manager.list_containers().await;

    docker_manager.remove_network(network_name).await;
    docker_manager.list_networks().await;
    
    docker_manager.remove_all_images_by_name(image_name).await;
    docker_manager.list_images().await;
} 