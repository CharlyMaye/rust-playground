use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct DockerCompose {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub services: HashMap<String, Service>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub networks: Option<HashMap<String, Network>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volumes: Option<HashMap<String, Volume>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Service {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build: Option<BuildConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ports: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub networks: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volumes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depends_on: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BuildConfig {
    pub context: String,
    pub dockerfile: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Network {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Volume {}

pub struct DockerComposeManager {
    compose_file: String,
}

impl DockerComposeManager {
    pub fn new(compose_file: &str) -> Self {
        Self {
            compose_file: compose_file.to_string(),
        }
    }

    /// Lire le fichier docker-compose.yml
    pub fn read(&self) -> Result<DockerCompose, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(&self.compose_file)?;
        let compose: DockerCompose = serde_yaml::from_str(&content)?;
        Ok(compose)
    }

    /// Écrire dans le fichier docker-compose.yml
    pub fn write(&self, compose: &DockerCompose) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yaml::to_string(compose)?;
        fs::write(&self.compose_file, yaml)?;
        Ok(())
    }

    /// Ajouter un service dynamiquement
    pub fn add_service(
        &self,
        service_name: &str,
        service: Service,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut compose = self.read()?;
        compose.services.insert(service_name.to_string(), service);
        self.write(&compose)?;
        println!("✅ Service '{}' ajouté au docker-compose.yml", service_name);
        Ok(())
    }

    /// Supprimer un service
    pub fn remove_service(&self, service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut compose = self.read()?;
        compose.services.remove(service_name);
        self.write(&compose)?;
        println!("✅ Service '{}' supprimé du docker-compose.yml", service_name);
        Ok(())
    }

    /// Ajouter un réseau
    pub fn add_network(&self, network_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut compose = self.read()?;
        let networks = compose.networks.get_or_insert_with(HashMap::new);
        networks.insert(
            network_name.to_string(),
            Network {
                driver: Some("bridge".to_string()),
            },
        );
        self.write(&compose)?;
        println!("✅ Réseau '{}' ajouté au docker-compose.yml", network_name);
        Ok(())
    }

    /// Ajouter un volume
    pub fn add_volume(&self, volume_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut compose = self.read()?;
        let volumes = compose.volumes.get_or_insert_with(HashMap::new);
        volumes.insert(volume_name.to_string(), Volume {});
        self.write(&compose)?;
        println!("✅ Volume '{}' ajouté au docker-compose.yml", volume_name);
        Ok(())
    }

    /// Lister tous les services
    pub fn list_services(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let compose = self.read()?;
        let services: Vec<String> = compose.services.keys().cloned().collect();
        println!("📋 Services dans docker-compose.yml:");
        for service in &services {
            println!("  - {}", service);
        }
        Ok(services)
    }

    /// Démarrer les services (docker-compose up)
    pub fn up(&self, detached: bool) -> Result<(), String> {
        let mut args = vec!["-f", &self.compose_file, "up"];
        if detached {
            args.push("-d");
        }

        let output = Command::new("docker-compose")
            .args(&args)
            .output()
            .map_err(|e| e.to_string())?;

        if output.status.success() {
            println!("✅ docker-compose up exécuté avec succès");
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    /// Arrêter les services (docker-compose down)
    pub fn down(&self) -> Result<(), String> {
        let output = Command::new("docker-compose")
            .args(["-f", &self.compose_file, "down"])
            .output()
            .map_err(|e| e.to_string())?;

        if output.status.success() {
            println!("✅ docker-compose down exécuté avec succès");
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    /// Redémarrer un service spécifique
    pub fn restart_service(&self, service_name: &str) -> Result<(), String> {
        let output = Command::new("docker-compose")
            .args(["-f", &self.compose_file, "restart", service_name])
            .output()
            .map_err(|e| e.to_string())?;

        if output.status.success() {
            println!("✅ Service '{}' redémarré", service_name);
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    /// Démarrer un service spécifique
    pub fn start_service(&self, service_name: &str) -> Result<(), String> {
        let output = Command::new("docker-compose")
            .args(["-f", &self.compose_file, "start", service_name])
            .output()
            .map_err(|e| e.to_string())?;

        if output.status.success() {
            println!("✅ Service '{}' démarré", service_name);
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    /// Arrêter un service spécifique
    pub fn stop_service(&self, service_name: &str) -> Result<(), String> {
        let output = Command::new("docker-compose")
            .args(["-f", &self.compose_file, "stop", service_name])
            .output()
            .map_err(|e| e.to_string())?;

        if output.status.success() {
            println!("✅ Service '{}' arrêté", service_name);
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }
}