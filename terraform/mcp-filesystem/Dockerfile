FROM node:20-alpine

# Pre-install packages to avoid npx cold start on Cloud Run
WORKDIR /app
RUN npm init -y && npm install mcp-proxy @modelcontextprotocol/server-filesystem

WORKDIR /projects
RUN mkdir -p reservation_status

EXPOSE 8080

# Use PORT from Cloud Run (default 8080). npx uses pre-installed node_modules.
CMD ["sh", "-c", "cd /app && npx mcp-proxy --port ${PORT:-8080} --server stream --streamEndpoint /mcp --stateless -- npx @modelcontextprotocol/server-filesystem /projects/reservation_status"]
