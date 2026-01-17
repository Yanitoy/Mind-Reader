const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 3000;
const FRONTEND_ROOT = path.join(__dirname, "..", "frontend");
const DIST_DIR = path.join(FRONTEND_ROOT, "dist");
const FRONTEND_DIR = fs.existsSync(DIST_DIR) ? DIST_DIR : FRONTEND_ROOT;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml"
};

function send(res, statusCode, body, contentType = "text/plain; charset=utf-8") {
  res.writeHead(statusCode, { "Content-Type": contentType });
  res.end(body);
}

function serveFile(res, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const type = MIME_TYPES[ext] || "application/octet-stream";

  fs.readFile(filePath, (err, data) => {
    if (err) {
      send(res, 500, "Server error.");
      return;
    }
    send(res, 200, data, type);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let pathname = decodeURIComponent(url.pathname);

  if (pathname === "/") {
    pathname = "/index.html";
  }

  const filePath = path.join(FRONTEND_DIR, pathname);
  if (!filePath.startsWith(FRONTEND_DIR)) {
    send(res, 403, "Forbidden.");
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err) {
      send(res, 404, "Not found.");
      return;
    }

    if (stats.isDirectory()) {
      const indexPath = path.join(filePath, "index.html");
      fs.stat(indexPath, (indexErr) => {
        if (indexErr) {
          send(res, 404, "Not found.");
          return;
        }
        serveFile(res, indexPath);
      });
      return;
    }

    serveFile(res, filePath);
  });
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Mind Reader server running at http://localhost:${PORT}`);
});
