use std::collections::HashMap;

use bollard::Docker;
use bollard::container::{Config, CreateContainerOptions, ListContainersOptions, RemoveContainerOptions, StartContainerOptions};
use bollard::image::{CreateImageOptions, ListImagesOptions};
use futures::stream::StreamExt;

// // Gestion des conteneurs
// docker.create_container(...)
// docker.start_container(...)
// docker.stop_container(...)
// docker.remove_container(...)

// // Images
// docker.list_images(...)
// docker.create_image(...)
// docker.remove_image(...)

// // Volumes
// docker.list_volumes(...)
// docker.create_volume(...)

// // Réseaux
// docker.list_networks(...)
// docker.create_network(...)

// // Logs et stats
// docker.logs(...)
// docker.stats(...)

pub struct DockerManager {
    docker: Docker,
}

pub trait DockerImageManager {
    async fn list_images(&self);
    async fn create_image(&self, image_name: &str);
    async fn remove_all_images_by_name(&self, image_name: &str) ;
}

impl DockerManager {
    pub fn new() -> Self {
        DockerManager {
            docker: Docker::connect_with_local_defaults().unwrap(),
        }
    }
}
// Info
impl DockerManager {
    pub async fn docker_info(&self) {
        println!("\n🐳 Version Docker:");
        let version = self.docker.version().await.unwrap();
        println!("  Version: {}", version.version.unwrap_or_default());
        println!("  API Version: {}", version.api_version.unwrap_or_default());
    }
}
// Gestion des images
impl DockerManager {
    async fn remove_image(&self, image_name: &str) {
        match self.docker.remove_image(image_name, None, None).await {
            Ok(_) => println!("Image {} removed successfully", image_name),
            Err(e) => eprintln!("Error removing image {}: {}", image_name, e),
        }
    }
}
impl DockerImageManager for DockerManager {
    async fn list_images(&self) {
        let filters: HashMap<&str, Vec<&str>> = HashMap::new();
        // filters.insert("dangling", vec!["true"]);

        let options = Some(ListImagesOptions{
        all: true,
        filters,
        ..Default::default()
        });

        let result = self.docker.list_images(options).await.unwrap();
        for image in result {
            println!("Image ID: {}", image.id);
        }
    }
    
    async fn create_image(&self, image_name: &str) {
        let options = Some(CreateImageOptions{
        from_image: image_name,
        ..Default::default()
        });

        let stream = self.docker.create_image(options, None, None);
        tokio::pin!(stream);
        while let Some(progress) = stream.next().await {
            match progress {
                Ok(output) => println!("{:?}", output),
                Err(e) => {
                    // Ignorer les erreurs de fin de stream
                    if !e.to_string().contains("stream") {
                        eprintln!("Error: {}", e);
                    }
                }
            }
        }
    }

    async fn remove_all_images_by_name(&self, image_name: &str) {
        // Lister toutes les images
        let images = self.docker.list_images(Some(ListImagesOptions::<String> {
            all: true,
            ..Default::default()
        })).await.unwrap();

        // Trouver et supprimer toutes les images qui contiennent le nom
        for tag in images.iter()
            .flat_map(|img| &img.repo_tags)
            .filter(|tag| tag.starts_with(image_name)) 
        {
            self.remove_image(tag).await;
        }
    }
}
// Gestion des conteneurs
impl DockerManager {
    pub async fn create_container(&self, image_name: &str, container_name: &str) -> String {
        let options = Some(CreateContainerOptions{
            name: container_name,
            platform: None,
        });

        let config = Config {
            image: Some(image_name),
            // cmd: Some(vec!["/hello"]),
            ..Default::default()
        };

        let result = self.docker.create_container(options, config).await.unwrap();
        println!("Container created with ID: {}", result.id);
        result.id
    }
    pub async fn remove_container(&self, container_id: &str) {
        let options = Some(RemoveContainerOptions{
            force: true,
            ..Default::default()
        });

        self.docker.remove_container(container_id, options).await.unwrap();
    }
    pub async fn list_containers(&self) {
        println!("📦 Liste des conteneurs Docker:");
        let containers = self.docker.list_containers(Some(ListContainersOptions::<String> {
            all: true,
            ..Default::default()
        })).await.unwrap();
        
        for container in containers {
            let id = container.id.as_ref().map(|s| &s[..12]).unwrap_or("N/A");
            let names = container.names.unwrap_or_default().join(", ");
            let image = container.image.unwrap_or_default();
            let state = container.state.unwrap_or_default();
            
            println!("  • {} | {} | {} | {}", id, names, image, state);
        }
    }

    pub async fn start_container(&self, container_id: &str) {
        self.docker.start_container(container_id, None::<StartContainerOptions<String>>).await.unwrap();
    }
    pub async fn stop_container(&self, container_id: &str) {
        self.docker.stop_container(container_id, None).await.unwrap();
    }
}

