from flask import request, Response, Flask
import requests

app = Flask("__main__")
SITE_NAME = "https://crocodile-gqhfy6c73a-uc.a.run.app/"


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>", methods=["GET", "POST", "PUT"])
def proxy(path):
    print(f"Received request: {path}")
    url = request.url.replace(request.host_url, f"{SITE_NAME}")
    print(f"Redirecting to: {url}")
    res = requests.request(  # ref. https://stackoverflow.com/a/36601467/248616
        method=request.method,
        url=url,
        headers={
            k: v for k, v in request.headers if k.lower() != "host"
        },  # exclude 'host' header
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
    )

    # region exlcude some keys in :res response
    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]  # NOTE we here exclude all "hop-by-hop headers" defined by RFC 2616 section 13.5.1 ref. https://www.rfc-editor.org/rfc/rfc2616#section-13.5.1
    headers = [
        (k, v) for k, v in res.raw.headers.items() if k.lower() not in excluded_headers
    ]
    # endregion exlcude some keys in :res response

    response = Response(res.content, res.status_code, headers)
    return response
