import cv2
import numpy as np
import torch
from pathlib import Path

from src.model.face_reconstruction import FaceReconstructionModel

from insightface.app import FaceAnalysis

from src.evaluation.similarity import cosine_similarity
from src.config import EMBEDDING_DIR

app = FaceAnalysis(name="buffalo_s")

app.prepare(
    ctx_id=0,
    det_size=(640,640),
)

def tensor_to_image(tensor):

    image = tensor.detach().cpu()

    image = image.permute(1,2,0).numpy()

    image = np.clip(image,0,1)

    image = (image*255).astype(np.uint8)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    return image
def extract_embedding(image):

    faces = app.get(image)

    if len(faces) == 0:
        return None

    return faces[0].embedding
def load_model(checkpoint_path):

    model = FaceReconstructionModel()

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(state)

    model.eval()

    return model
def calculate_identity_score(
        original_embedding,
        generated_image
):

    generated_embedding = extract_embedding(
        generated_image
    )

    if generated_embedding is None:
        return 0.0


    return cosine_similarity(
        original_embedding,
        generated_embedding
    )

def detect_face(image):

    faces = app.get(image)

    print(
        "Detected faces:",
        len(faces)
    )

    return faces

if __name__ == "__main__":

    model = load_model(
        "outputs/checkpoints/model_epoch_010.pth"
    )

    original_embedding = np.load(
        EMBEDDING_DIR / "000001.npy"
    ).astype(np.float32)


    input_embedding = torch.from_numpy(
        original_embedding
    ).unsqueeze(0)


    with torch.no_grad():

        generated = model(
            input_embedding
        )[0]


    image = tensor_to_image(
        generated
    )


    cv2.imwrite(
        "generated.jpg",
        image
    )

    detect_face(image)

    score = calculate_identity_score(
        original_embedding,
        image
    )


    print(
        "Generated image saved."
    )

    print(
        "Identity Score:",
        score
    )