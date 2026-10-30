"""FastAPI application and HTTP routes."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build the FastAPI app.

    A factory (rather than a module-level app) keeps tests free to build a
    fresh instance with their own settings later on.
    """
    app = FastAPI(title="cited-answers")

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness only for now; a database ping will be added once the DB layer exists.
        return {"status": "ok"}

    return app


app = create_app()
