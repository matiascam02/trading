# LLM Agent Trader

AI-powered stock trading backtesting system that integrates Large Language Models for intelligent trading decision analysis.

## System Architecture

```mermaid
flowchart TD
    %% User Interface Layer
    A[Frontend - Next.js] --> B[API Gateway - FastAPI Backend]
    
    %% Main Function Modules
    B --> C[LLM Streaming Backtest Engine]
    B --> D[Backtest Analysis API]
    B --> E[Daily Feedback API]
    
    %% Data Layer
    F[Stock Data Service<br/>YFinance] --> C
    G[SQLite Database<br/>Backtest Logs] --> D
    G --> E
    
    %% LLM Strategy Engine
    C --> H[LLM Smart Strategy]
    H --> I[Azure OpenAI<br/>GPT-4]
    H --> J[Technical Analysis Engine]
    H --> K[Risk Management Module]
    
    %% Backtest Execution Flow
    C --> L[Trading Signal Generation]
    L --> M[Performance Calculation]
    M --> N[Result Recording]
    N --> G
    
    %% Style Definitions
    classDef frontend fill:#e1f5fe
    classDef backend fill:#f3e5f5
    classDef llm fill:#fff3e0
    classDef data fill:#e8f5e8
    
    class A frontend
    class B,C,D,E backend
    class H,I,J,K llm
    class F,G data
```

## Quick Start

### Prerequisites
- **macOS/Linux**: Native support for `make` commands
- **Windows**: May require additional setup (WSL, Git Bash, or make utility installation)

### Install Dependencies

```bash
make install
```

### Environment Setup
Copy and configure your `.env` file:
```bash
cp .env.example .env
```

**Provider Selection (Auto-detect):** The backend picks the first available provider by key presence:
1) Azure OpenAI → `AZURE_OPENAI_API_KEY`
2) OpenAI (Direct) → `OPENAI_API_KEY`
3) Google Gemini → `GOOGLE_API_KEY`

Comment out keys to disable a provider.

Example configurations:

```env
# ----- Use OpenAI (direct) -----
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o  # can be gpt-4o-mini, or gpt-5 when available
```

```env
# ----- Use Azure OpenAI -----
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
```

```env
# ----- Use Google Gemini -----
GOOGLE_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
```

### Development Mode
```bash
make run
```

**🎉 Success!** After setup, open your browser and navigate to:
**http://localhost:3000** to access the web application

### Other Commands
```bash
make stop     # Stop all services
make test     # Run tests
make clean    # Clean cache files
make format   # Format code
```

### Windows Users
If you encounter issues with `make` commands on Windows, consider:
- **WSL (Windows Subsystem for Linux)**: Recommended approach
- **Git Bash**: Included with Git for Windows
- **Make for Windows**: Install GNU Make utility
