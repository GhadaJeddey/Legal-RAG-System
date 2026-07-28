# Pour lancer l'interface, il faut : 

1. Backend — depuis backend/inference_pipeline/ :

cd backend/inference_pipeline
source ../venv/bin/activate
uvicorn api:app --reload --port 8000
Vérifie aussi que Postgres tourne : docker compose up -d depuis la racine du projet si le conteneur n'est pas déjà lancé.

2. Frontend — depuis frontend/, dans un autre terminal :

cd frontend
npm run dev