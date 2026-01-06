FROM python:3.13-trixie

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY --chown=user . /home/user/app/

WORKDIR /home/user/app
RUN uv sync
ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

CMD ["make", "leaderboard"]