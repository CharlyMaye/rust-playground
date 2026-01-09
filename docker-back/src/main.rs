mod docker;


use docker::{DockerManager, DockerImageManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sleep_time = std::time::Duration::from_secs(1);
    let image_name = "hello-world";

    let docker_manager = DockerManager::new();
    docker_manager.docker_info().await;

    docker_manager.create_image(image_name).await;
    
    let container_id = docker_manager.create_container(image_name, "my_container").await;
    tokio::time::sleep(sleep_time).await;
    docker_manager.list_containers().await;

    docker_manager.start_container(&container_id).await;
    tokio::time::sleep(sleep_time).await;

    docker_manager.stop_container(&container_id).await;
    tokio::time::sleep(sleep_time).await;

    docker_manager.remove_container(&container_id).await;
    tokio::time::sleep(sleep_time).await;

    docker_manager.list_containers().await;

    docker_manager.remove_all_images_by_name(image_name).await;
    docker_manager.list_images().await;

    Ok(())
}
