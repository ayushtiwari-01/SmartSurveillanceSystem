from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


class FaceNetRecognizer:
    """Face recognition using FaceNet (InceptionResnetV1) and MTCNN."""

    def __init__(self, image_size: int = 160, match_threshold: float = 0.75) -> None:
        self.image_size = image_size
        self.match_threshold = match_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(image_size=image_size, keep_all=False, device=self.device)
        
        # Initialize FaceNet model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.known_labels: List[str] = []
        self.known_embeddings: Optional[np.ndarray] = None

    def set_known(self, labels: List[str], embeddings: Optional[np.ndarray]) -> None:
        """Set known person labels and embeddings."""
        self.known_labels = labels
        self.known_embeddings = embeddings.astype(np.float32) if embeddings is not None else None

    def compute_embedding(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Detect face and compute embedding.
        
        Returns:
            (embedding, face_box) or (None, None)
        """
        # Detect face
        boxes, _ = self.mtcnn.detect(image)
        
        if boxes is None or len(boxes) == 0:
            return None, None
        
        # Get largest face
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Extract and align face
        face_tensor = self.mtcnn(image)
        
        if face_tensor is None:
            return None, None
        
        # Compute embedding
        with torch.no_grad():
            embedding = self.model(face_tensor.unsqueeze(0).to(self.device))
        
        embedding_np = embedding.cpu().numpy().flatten()
        
        return embedding_np, (x1, y1, x2, y2)

    def match(self, embedding: np.ndarray) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """Match an embedding against known embeddings.
        
        Returns:
            (person_id, name, similarity_score) or (None, None, None)
        """
        if self.known_embeddings is None or len(self.known_embeddings) == 0:
            return None, None, None
        
        # Compute cosine similarity
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        known_norm = self.known_embeddings / (np.linalg.norm(self.known_embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(known_norm, emb_norm)
        
        idx = int(np.argmax(similarities))
        score = float(similarities[idx])
        
        if score >= self.match_threshold:
            name = self.known_labels[idx]
            return idx, name, score
        
        return None, None, None
