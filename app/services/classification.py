# app/services/classification.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights
import timm
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

from typing import Optional, List
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from app.config import settings

logger = logging.getLogger(__name__)

class COCOPretrainedClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, freeze_backbone=False):
        super(COCOPretrainedClassifier, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif model_name == 'resnet101':
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'mobilenet_v3_large':
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class ClassificationService:
    def __init__(self):
        """Initialize classification service using config settings"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = settings.CLASSIFICATION_MODEL_PATH
        self.conf_threshold = settings.CLASSIFICATION_CONFIDENCE_THRESHOLD
        
        # Load model metadata
        self._load_model_metadata()
        
        # Initialize and load model
        self._initialize_model()
        
        # Setup transforms
        self._setup_transforms()

    def _load_model_metadata(self):
        """Load model metadata from checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.num_classes = len(checkpoint.get('class_names', settings.DEFAULT_CLASSIFICATION_CLASSES))
            self.class_names = checkpoint.get('class_names', settings.DEFAULT_CLASSIFICATION_CLASSES)
            self.model_name = checkpoint.get('model_name', settings.DEFAULT_CLASSIFICATION_MODEL_BACKBONE)
            # logger.info(f"Loaded model metadata: {self.num_classes} classes, backbone: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}. Using defaults.")
            self.num_classes = len(settings.DEFAULT_CLASSIFICATION_CLASSES)
            self.class_names = settings.DEFAULT_CLASSIFICATION_CLASSES
            self.model_name = settings.DEFAULT_CLASSIFICATION_MODEL_BACKBONE

    def _initialize_model(self):
        """Initialize and load the model"""
        # logger.info(f"Initializing {self.model_name} classifier...")
        
        self.model = COCOPretrainedClassifier(
            num_classes=self.num_classes,
            model_name=self.model_name,
            pretrained=False
        )
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            # logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load classification model: {e}")

        self.model.to(self.device)
        self.model.eval()

    def _setup_transforms(self):
        """Setup image transforms"""
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def classify_card(self, image: np.ndarray) -> Optional[str]:
        """
        Classify card type from image
        
        Args:
            image: Input image array (OpenCV BGR format)
            
        Returns:
            Predicted card type or None if confidence below threshold
        """
        self.model.eval()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # Apply transforms
            transformed = self.transform(image=image_rgb)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)
                
                confidence = max_prob.item()
                predicted_class_idx = predicted_idx.item()

            if confidence >= self.conf_threshold:
                predicted_class = self.class_names[predicted_class_idx]
                logger.info(f"Classified as: {predicted_class} (confidence: {confidence:.2f})")
                return predicted_class
            
            logger.warning(f"Classification confidence {confidence:.2f} below threshold {self.conf_threshold}")
            return None
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return None