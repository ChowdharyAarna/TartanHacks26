vm: source .venv/bin/activate
backend: uvicorn app:app --reload --port 8000
frontend: 
npm install
npm run dev