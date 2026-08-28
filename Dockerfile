# Builds the hosted HTTP server (serve.mjs). The npm package itself needs no
# Docker; this exists for the Render deployment described in render.yaml.
FROM node:22-alpine
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

ENV NODE_ENV=production
EXPOSE 10000
CMD ["node", "serve.mjs"]
