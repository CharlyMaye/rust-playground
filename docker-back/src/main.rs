mod docker;

use bollard::Docker;
use bollard::container::ListContainersOptions;

use docker::DockerManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connexion à Docker
    let docker = Docker::connect_with_local_defaults()?;
    

    
    // Informations système Docker
    println!("\n🐳 Version Docker:");
    let version = docker.version().await?;
    println!("  Version: {}", version.version.unwrap_or_default());
    println!("  API Version: {}", version.api_version.unwrap_or_default());

    let docker_manager = DockerManager::new();
    docker_manager.create_image("hello-world").await;
    let container_id = docker_manager.create_container("hello-world", "my_container").await;

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    docker_manager.list_containers().await;

    docker_manager.remove_container(&container_id).await;
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    docker_manager.list_containers().await;

    docker_manager.remove_all_images_by_name("hello-world").await;
    docker_manager.list_images().await;

    Ok(())
}
