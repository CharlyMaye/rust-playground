mod docker;
mod docker_compose;

use docker_compose::{DockerComposeManager, Service};
use std::collections::HashMap;

use docker::{DockerManager, DockerImageManager, DockerContainerManager, DockerNetworkManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {



    Ok(())
}

async fn test_docker_compose() {
        let compose_manager = DockerComposeManager::new("docker-compose.yml");

    // Créer un nouveau service
    let mut env = HashMap::new();
    env.insert("POSTGRES_PASSWORD".to_string(), "secret".to_string());

    let postgres_service = Service {
        image: "postgres:15".to_string(),
        container_name: Some("my-postgres".to_string()),
        ports: Some(vec!["5432:5432".to_string()]),
        environment: Some(env),
        networks: Some(vec!["my-network".to_string()]),
        volumes: Some(vec!["postgres-data:/var/lib/postgresql/data".to_string()]),
        depends_on: None,
    };

    // Ajouter le service
    compose_manager.add_service("postgres", postgres_service).unwrap();

    // Ajouter un réseau
    compose_manager.add_network("my-network").unwrap();

    // Démarrer les services
    compose_manager.up(true).unwrap();

    // Supprimer un service
    // compose_manager.remove_service("postgres").unwrap();

    // Arrêter tous les services
    // compose_manager.down().unwrap();
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