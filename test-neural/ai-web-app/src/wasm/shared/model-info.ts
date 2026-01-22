export interface ModelInfo {
    name: string;
    architecture: string;
    accuracy: number;
    description: string;  
    test_samples: number;
    trained_at: Date;
}