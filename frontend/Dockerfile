# Vite inlines VITE_* env vars at BUILD time, not runtime - so the backend
# URL the built bundle will call is fixed here via a build arg, not the
# container's runtime environment.
ARG VITE_API_URL=http://localhost:8000

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
    CMD wget -q -O- http://127.0.0.1:80/health || exit 1
