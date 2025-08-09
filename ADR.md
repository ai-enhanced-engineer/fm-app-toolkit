# ADR-001: AI Base Template

## Status
Accepted

## Context
A minimal Python template for AI/ML projects that provides a standardized starting point with modern tooling, best practices, and pre-configured libraries for rapid prototyping and clean architecture.

## Decision
Build a lightweight, opinionated template that balances simplicity with completeness, providing essential ML/data science tools while maintaining flexibility for different project types.

## Consequences
- **Pros**: Faster project initialization, consistent structure across projects, reduced setup time, built-in best practices
- **Cons**: May include unnecessary dependencies for simple projects, opinionated choices may not suit all use cases

## Technical Specification
- **Stack**: Python 3.12, FastAPI, Pydantic, PyTorch, scikit-learn, XGBoost/LightGBM
- **API**: FastAPI-based REST API with automatic OpenAPI documentation
- **Dependencies**: uv for package management, pre-configured ML/data science libraries
- **Data Flow**: Input → FastAPI endpoints → Processing modules → Response models
- **State Management**: Stateless by default, configurable via environment variables
- **Scaling**: Horizontal scaling supported via FastAPI async capabilities

## Integration Points
- **Consumes**: Environment variables, .env files for configuration
- **Provides**: REST API endpoints, Jupyter notebook support for experimentation
- **Protocols**: HTTP/REST, async support via FastAPI

## Non-Functional Requirements
- **Performance**: Async request handling, efficient ML model serving
- **Availability**: Development/research focused, no specific SLA
- **Security**: Environment-based configuration, no hardcoded secrets
- **Scalability**: Suitable for prototypes to small/medium production deployments

## Deployment
- **Platform**: Local development, containerizable for K8s/cloud deployment
- **Configuration**: Environment variables via .env files, python-dotenv
- **Resources**: Depends on ML model complexity, baseline ~1GB RAM, 1 CPU core