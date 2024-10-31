import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
import subprocess
from transformers import BertTokenizer, BertForSequenceClassification
import json
import fire
import torch
from urllib.parse import urlparse, unquote


model: BertForSequenceClassification = None
tokenizer: BertTokenizer = None
device: str = None

def log(*args):
    print(f"[{os.environ.get('RANK', '')}]", *args, file=sys.stderr)


class RequestHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        query = unquote(urlparse(self.path).query)

        if not query:
            self.begin_content('text/html')

            html = os.path.join(os.path.dirname(__file__), 'index.html')
            self.wfile.write(open(html).read().encode())
            return

        self.begin_content('application/json;charset=UTF-8')

        tokens = tokenizer.encode(query)
        all_tokens = len(tokens)
        tokens = tokens[:tokenizer.model_max_length - 2]
        used_tokens = len(tokens)
        tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
        mask = torch.ones_like(tokens)

        with torch.no_grad():
            logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
            probs = logits.softmax(dim=-1)

        fake, real = probs.detach().cpu().flatten().numpy().tolist()

        self.wfile.write(json.dumps(dict(
            all_tokens=all_tokens,
            used_tokens=used_tokens,
            real_probability=real,
            fake_probability=fake
        )).encode())

    def begin_content(self, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def log_message(self, format, *args):
        log(format % args)


def serve_forever(server, model, tokenizer, device):
    log('Process has started; loading the models ...')
    globals()['models'] = model.to(device)
    globals()['tokenizer'] = tokenizer
    globals()['device'] = device

    log(f'Ready to serve at http://localhost:{server.server_address[1]}')
    server.serve_forever()


def main(checkpoint, port=8080, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model_path = 'D:/Content_Secu/task4/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 明确地设置 BOS 和 EOS 令牌为 CLS 和 SEP 令牌
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    print(f'Loading checkpoint from {checkpoint}')
    data = torch.load(checkpoint, map_location='cpu')

    model = BertForSequenceClassification.from_pretrained(model_path)
    model.load_state_dict(data['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    print(f'Starting HTTP server on port {port}', file=sys.stderr)
    server = HTTPServer(('0.0.0.0', port), RequestHandler)
    serve_forever(server, model, tokenizer, device)



if __name__ == '__main__':
    fire.Fire(main)
