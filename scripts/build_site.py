"""暗号化したダッシュボードを作る（2026年11月25日）。

J-Quants の利用条件（データを第三者が閲覧できる状態にしない）に合わせ、リポジトリを非公開にしたうえで、
ダッシュボードだけを別の公開リポジトリ（GitHub Pages）に置く。置くのは画面（index.html）と、暗号化したデータだけ。
合言葉（GitHub の secrets の DASH_PASS）を知っている人だけが、ブラウザの中で復号して見られる。合言葉はサーバーに送らない。

暗号：合言葉から PBKDF2-HMAC-SHA256（60万回）で鍵を作り、AES-256-GCM で暗号化する。
ファイルごとに gzip で縮めてから暗号化し、「12バイトの nonce ＋ 暗号文」を <元の名前>.enc として書く。
使い方：DASH_PASS=... python scripts/build_site.py <出力先>
"""
import gzip
import json
import os
import re
import shutil
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = Path(__file__).resolve().parent.parent
SALT = b"dividend-heist-board-v1"
ITER = 600_000
FILES = ["results.json", "portfolio_state.json", "portfolio_history.json", "health.json",
         "rotation_state.json", "rotation_history.json", "swing_state.json", "swing_history.json",
         "value_state.json", "value_history.json"]

JS = r"""
/* 2026年11月25日：暗号化したデータを、合言葉で復号して読む（build_site.py が差し込む） */
const DH_SALT=new TextEncoder().encode('dividend-heist-board-v1');
let DH_KEY=null, DH_ASKED=false;
function dhB64e(b){let s='';new Uint8Array(b).forEach(x=>s+=String.fromCharCode(x));return btoa(s);}
function dhB64d(s){const t=atob(s);const u=new Uint8Array(t.length);for(let i=0;i<t.length;i++)u[i]=t.charCodeAt(i);return u;}
async function dhOpen(path,key){
  const r=await fetch(path+'.enc',{cache:'no-cache'});if(!r.ok)return null;
  const b=new Uint8Array(await r.arrayBuffer());
  const pt=await crypto.subtle.decrypt({name:'AES-GCM',iv:b.slice(0,12)},key,b.slice(12));
  const txt=await new Response(new Blob([pt]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  return JSON.parse(txt);
}
async function dhCheck(key){try{const c=await dhOpen('data/check.json',key);return !!(c&&c.ok);}catch(e){return false;}}
async function dhKey(){
  if(DH_KEY)return DH_KEY;
  let raw=null;try{raw=localStorage.getItem('dh_key_v1');}catch(e){}
  if(raw){try{const k=await crypto.subtle.importKey('raw',dhB64d(raw),'AES-GCM',true,['decrypt']);if(await dhCheck(k)){DH_KEY=k;return k;}}catch(e){}}
  if(DH_ASKED)return null;
  DH_ASKED=true;
  for(;;){
    const p=prompt('合言葉を入れてください（この端末に覚えさせます）');
    if(p===null||p==='')return null;
    const base=await crypto.subtle.importKey('raw',new TextEncoder().encode(p),'PBKDF2',false,['deriveKey']);
    const k=await crypto.subtle.deriveKey({name:'PBKDF2',salt:DH_SALT,iterations:600000,hash:'SHA-256'},base,{name:'AES-GCM',length:256},true,['decrypt']);
    if(await dhCheck(k)){DH_KEY=k;try{localStorage.setItem('dh_key_v1',dhB64e(await crypto.subtle.exportKey('raw',k)));}catch(e){}return k;}
    alert('合言葉が違います');
  }
}
function dhForget(){try{localStorage.removeItem('dh_key_v1');}catch(e){}location.reload();}
async function loadJSON(path){
  try{const k=await dhKey();if(!k)return null;return await dhOpen(path,k);}catch(e){return null;}
}
"""


def key_from(passphrase: str) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=SALT, iterations=ITER)
    return kdf.derive(passphrase.encode("utf-8"))


def seal(aes: AESGCM, raw: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + aes.encrypt(nonce, gzip.compress(raw, mtime=0), None)


def build(out: Path, passphrase: str) -> int:
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "history").mkdir(parents=True)
    aes = AESGCM(key_from(passphrase))
    n = 0
    (out / "data" / "check.json.enc").write_bytes(seal(aes, json.dumps({"ok": True}).encode()))
    for name in FILES:
        p = ROOT / "data" / name
        if p.exists():
            (out / "data" / (name + ".enc")).write_bytes(seal(aes, p.read_bytes()))
            n += 1
    for p in sorted((ROOT / "data" / "history").glob("h*.json")):
        (out / "data" / "history" / (p.name + ".enc")).write_bytes(seal(aes, p.read_bytes()))
        n += 1
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    # 元の loadJSON（平文のファイルと公開サイトを読む）を、復号して読む形に置き換える
    pat = re.compile(r"const REMOTE_BASE='[^']*';\nasync function loadJSON\(path\)\{\n.*?\n  return null;\n\}\n", re.S)
    html, k = pat.subn(lambda _m: JS.lstrip("\n"), html, count=1)
    if k != 1:
        raise SystemExit("index.html の loadJSON が見つかりません（形が変わった可能性）")
    html = html.replace("<head>", '<head>\n<meta name="robots" content="noindex,nofollow">', 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    (out / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    return n


if __name__ == "__main__":
    pw = os.environ.get("DASH_PASS", "")
    if not pw:
        raise SystemExit("DASH_PASS がありません")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    print(f"暗号化したファイル：{build(out, pw)}")
