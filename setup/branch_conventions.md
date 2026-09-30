# Branch Conventions

- `main`: Protected production-ready code. Merges come exclusively from `dev`.
- `dev`: Integration branch. Must always build and pass tests.
- `feature/<workstream>/<short-description>`: Workstream feature branches (e.g., `feature/ws-2/search-bar-component`, `feature/ws-4/fastapi-routes`).

## Commit Guidelines
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Reference workstream ID in commit messages (e.g., `feat(ws-4): implement mock embedding service`).
