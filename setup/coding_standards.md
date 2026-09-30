# Coding Standards & Guidelines

## Python (Backend & Data-Handling)
1. **Type Annotations**: Mandatory for all function signatures and public APIs. Run `mypy` before submitting PRs.
2. **Formatting**: Enforced via `ruff format` and `ruff check`.
3. **No Hidden Exceptions**: Catch specific exceptions (`ValueError`, `FileNotFoundError`, `rasterio.RasterioIOError`), never bare `except:`.
4. **Interfaces First**: Services must consume abstract base classes (`EmbeddingModel`, `ChangeDetector`, `DatasetAdapter`), never concrete implementations directly.

## TypeScript (Frontend)
1. **No `any`**: Use explicit interfaces or generics.
2. **Styling**: Vanilla CSS / CSS Modules only. Do not introduce utility CSS frameworks like Tailwind unless explicitly instructed.
3. **API Contracts**: Keep `frontend/src/types/api.types.ts` strictly synchronized with backend Pydantic schemas in `backend/app/schemas/`.
4. **API Isolation**: All HTTP calls must go through functions exported in `frontend/src/api/`. Components must not call `axios` directly.
