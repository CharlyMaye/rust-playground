mod docker;


use docker::DockerManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docker_manager = DockerManager::new();
    docker_manager.docker_info().await;

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
