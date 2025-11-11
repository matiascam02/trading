# Installation commands
.venv:
	@echo "Creating Python virtual environment (.venv) with uv..."
	@uv venv --python 3.12
	@echo "Virtual environment created at .venv"

install: .venv
	@echo "Installing dependencies..."
	@echo "Installing Python dependencies into .venv..."
	@uv sync
	@echo "Installing Node.js dependencies..."
	@cd frontend && npm install
	@echo "Installation complete!"
# Run the entire application (backend + frontend)
run:
	@echo "Starting LLM Agent Trader..."
	@echo "Starting backend server..."
	@uv run uvicorn app.main:app --app-dir backend --reload --reload-exclude '.venv/*' --host 0.0.0.0 --port 8000 &
	@sleep 3
	@echo "Starting frontend server..."
	@cd frontend && npm run dev &
	@echo "Backend running at: http://localhost:8000"
	@echo "Frontend running at: http://localhost:3000"
	@echo "API docs available at: http://localhost:8000/docs"
	@echo "Press Ctrl+C to stop all services"
	@wait

# Stop all services
stop:
	@echo "Stopping LLM Agent Trader services..."
	@echo "Killing processes on port 8000 (backend)..."
	@-lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No backend process found on port 8000"
	@echo "Killing processes on port 3000 (frontend)..."
	@-lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "No frontend process found on port 3000"
	@echo "All services stopped"

format:
	uv run ruff format -v .

lint:
	uv run ruff check --select I --fix .	

check:
	uv run pyright

unit-test:
	uv run pytest 

integration-test:
	uv run ./test/main.py

clean:
	@echo "Cleaning cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@cd frontend && rm -rf .next node_modules/.cache 2>/dev/null || true
	@echo "Clean complete!"
