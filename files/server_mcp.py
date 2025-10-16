# -*- coding: utf-8 -*-
# ==============================================
# server_mcp.py — MCP Server (FastMCP) for OCI
# ==============================================
# - Tool to create OCI resources via `oci` CLI (VM example + generic passthrough)
# --------------------------------------------------------------
import re
import shlex
import subprocess
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import configparser
from mcp.server.fastmcp import FastMCP

# Config File
with open("./config", "r") as f:
    config_data = json.load(f)

# FastMCP Server
mcp = FastMCP("oci-ops")

# ------------------------------
# Save Text Log
# ------------------------------
def append_line(file_path: str, base: list):
    """
    Save the sequence of commands in `base` to a text file.

    Args:
        file_path (str): Path to the text file.
        base (list): List of command parts to save.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        # join each item in base with a space if it's a command string
        command_line = " ".join(map(str, base))
        f.write(command_line + "\n")
        f.flush()

# ------------------------------
# OCI CLI execution helper
# ------------------------------

class OCI:
    def __init__(self, profile: Optional[str] = None, bin_path: Optional[str] = None):
        self.profile = config_data["oci_profile"]
        self.bin = config_data["OCI_CLI_BIN"]

    def run(self, args: List[str]) -> Tuple[int, str, str]:
        try:
            base = [self.bin]
            if self.profile:
                base += ["--profile", self.profile]
            cmd = base + args
            append_line("log.txt", cmd)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            append_line("log.txt", proc.stdout)
            append_line("log.txt", proc.stderr)
            append_line("log.txt", "--------------------------")
            return proc.returncode, proc.stdout, proc.stderr
        except ex as Exception:
            append_line("log.txt", str(ex))

oci_cli = OCI(profile=config_data["oci_profile"])

# -------- OCI config helpers --------
def _read_oci_config(profile: Optional[str]) -> Dict[str, str]:
    cfg_path = os.path.expanduser("~/.oci/config")
    cp = configparser.ConfigParser()
    if os.path.exists(cfg_path):
        cp.read(cfg_path)
        prof = config_data["oci_profile"]
        if cp.has_section(prof):
            return {k: v for k, v in cp.items(prof)}
    return {}

def _tenancy_ocid() -> Optional[str]:
    return _read_oci_config(config_data["oci_profile"]).get("tenancy")

def _safe_json(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return {"raw": s}

# ---------------------------------
# Phonetic + fuzzy helpers (pt-BR)
# ---------------------------------
_consonant_map = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def ptbr_soundex(word: str, maxlen: int = 6) -> str:
    w = _normalize(word)
    if not w:
        return ""
    first_letter = w[0]
    # Remove vowels and h/w/y after first letter, collapse duplicates
    digits = []
    prev = ""
    for ch in w[1:]:
        if ch in "aeiouhwy ":
            code = ""
        else:
            code = _consonant_map.get(ch, "")
        if code and code != prev:
            digits.append(code)
        prev = code
    code = (first_letter + "".join(digits))[:maxlen]
    return code.ljust(maxlen, "0")

from difflib import SequenceMatcher

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

# ------------------------------
# MCP Tools
# ------------------------------
@mcp.tool()
async def find_subnet(query_text: str) -> dict:
    """
    Find the subnet ocid by the name and the compartment ocid
    """
    structured = f"query subnet resources where displayName =~ '.*{query_text}*.'"
    code, out, err = oci_cli.run(["search","resource","structured-search","--query-text", structured])
    if code != 0:
        return {"status":"error","stderr": err, "stdout": out}
    data = json.loads(out)
    items = data.get("data",{}).get("items",[])
    return {"status":"ok","data": items}

@mcp.tool()
async def list_availability_domains(compartment_ocid: Optional[str] = None) -> Dict[str, Any]:
    """List ADs with `oci iam availability-domain list`."""
    cid = compartment_ocid or _tenancy_ocid()
    if not cid:
        return {"status": "error", "error": "Missing tenancy compartment OCID."}
    code, out, err = oci_cli.run(["iam", "availability-domain", "list", "--compartment-id", cid])
    if code != 0:
        return {"status": "error", "stderr": err, "stdout": out}
    return {"status": "ok", "data": _safe_json(out)}

@mcp.tool()
async def find_ad(name_or_hint: str, compartment_ocid: Optional[str] = None) -> Dict[str, Any]:
    """Find the AD by a name (ex.: 'SAOPAULO-1-AD-1')."""
    lst = await list_availability_domains(compartment_ocid)
    if lst.get("status") != "ok":
        return lst
    items = lst["data"].get("data", []) if isinstance(lst["data"], dict) else []
    q = _normalize(name_or_hint)
    scored = []
    for ad in items:
        adname = ad.get("name") or ad.get("display-name") or ""
        s = similarity(q, adname)
        scored.append((s, adname))
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored:
        return {"status": "not_found", "candidates": []}
    best = scored[0]
    return {"status": "ok" if best[0] >= 0.6 else "ambiguous", "ad": scored[0][1], "candidates": [n for _, n in scored[:5]]}

async def list_shapes(compartment_ocid: Optional[str] = None, ad: Optional[str] = None) -> Dict[str, Any]:
    """List the shapes with `oci compute shape list --all` (needs compartment; AD is optional)."""
    cid = compartment_ocid or _tenancy_ocid()
    if not cid:
        return {"status": "error", "error": "Missing compartment OCID."}
    args = ["compute", "shape", "list", "--compartment-id", cid, "--all"]
    if ad:
        args += ["--availability-domain", ad]
    code, out, err = oci_cli.run(args)
    if code != 0:
        return {"status": "error", "stderr": err, "stdout": out}
    data = _safe_json(out)
    return {"status": "ok", "data": data.get("data", []) if isinstance(data, dict) else data}

@mcp.tool()
async def resolve_shape(hint: str, compartment_ocid: Optional[str] = None, ad: Optional[str] = None) -> Dict[str, Any]:
    """Resolve shape informing a name 'e4' → find all shapes have e4 like 'VM.Standard.E4.Flex'."""
    lst = await list_shapes(compartment_ocid=compartment_ocid, ad=ad)
    if lst.get("status") != "ok":
        return lst
    items = lst["data"]
    q = _normalize(hint)
    scored = []
    for s in items:
        name = s.get("shape") or ""
        s1 = similarity(q, name)
        # bônus para begins-with no sufixo da família
        fam = _normalize(name.replace("VM.Standard.", ""))
        s1 += 0.2 if fam.startswith(q) or q in fam else 0
        scored.append((s1, name))
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored:
        return {"status": "not_found", "candidates": []}
    best = scored[0]
    return {"status": "ok" if best[0] >= 0.6 else "ambiguous", "shape": best[1], "candidates": [n for _, n in scored[:5]]}

async def list_images(compartment_ocid: Optional[str] = None,
                      operating_system: Optional[str] = None,
                      operating_system_version: Optional[str] = None,
                      shape: Optional[str] = None) -> Dict[str, Any]:
    """Find the image by a short name or similarity"""
    cid = compartment_ocid or _tenancy_ocid()
    if not cid:
        return {"status": "error", "error": "Missing compartment OCID."}
    args = ["compute", "image", "list", "--compartment-id", cid, "--all"]
    if operating_system:
        args += ["--operating-system", operating_system]
    if operating_system_version:
        args += ["--operating-system-version", operating_system_version]
    if shape:
        args += ["--shape", shape]
    code, out, err = oci_cli.run(args)
    if code != 0:
        return {"status": "error", "stderr": err, "stdout": out}
    data = _safe_json(out)
    items = data.get("data", []) if isinstance(data, dict) else []
    return {"status": "ok", "data": items}

@mcp.tool()
async def resolve_image(query: str,
                        compartment_ocid: Optional[str] = None,
                        shape: Optional[str] = None) -> Dict[str, Any]:
    """Find the image by a short name or similarity"""
    # heuristic
    q = query.strip()
    os_name, os_ver = None, None
    # examples: "Oracle Linux 9", "OracleLinux 9", "OL9"
    if "linux" in q.lower():
        os_name = "Oracle Linux"
        m = re.search(r"(?:^|\\D)(\\d{1,2})(?:\\D|$)", q)
        if m:
            os_ver = m.group(1)

    # Filter for version
    lst = await list_images(compartment_ocid=compartment_ocid, operating_system=os_name, operating_system_version=os_ver)
    if lst.get("status") != "ok":
        return lst
    items = lst["data"]
    if not items:
        # fallback: sem filtro, listar tudo e fazer fuzzy no display-name
        lst = await list_images(compartment_ocid=compartment_ocid)
        if lst.get("status") != "ok":
            return lst
        items = lst["data"]

    # ranking for display-name and creation date
    ranked = []
    for img in items:
        dn = img.get("display-name","")
        s = similarity(query, dn)
        ts = img.get("time-created") or img.get("time_created") or ""
        ranked.append((s, ts, img))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if not ranked:
        return {"status": "not_found", "candidates": []}

    best = ranked[0][2]
    # top-5 candidates
    cands = []
    for s, _, img in ranked[:5]:
        cands.append({"name": img.get("display-name"), "ocid": img["id"], "score": round(float(s), 4)})

    status = "ok" if cands and cands[0]["score"] >= 0.65 else "ambiguous"
    return {"status": status, "resource": cands[0] if cands else None, "candidates": cands}

def _norm(s: str) -> str:
    return _normalize(s)

@mcp.tool()
async def find_compartment(query_text: str) -> dict:
    """
        Find compartment ocid by the name, the compartment ocid is the identifier field
    """
    structured = f"query compartment resources where displayName =~ '.*{query_text}*.'"
    code, out, err = oci_cli.run(["search","resource","structured-search","--query-text", structured])
    if code != 0:
        return {"status":"error","stderr": err, "stdout": out}
    data = json.loads(out)
    items = data.get("data",{}).get("items",[])
    return {"status":"ok","data": items}

@mcp.tool()
async def create_compute_instance(
        compartment_ocid: Optional[str] = None,
        subnet_ocid: Optional[str] = None,
        availability_domain: Optional[str] = None,
        shape: Optional[str] = None,
        ocpus: Optional[int] = None,       # Inteiro opcional
        memory: Optional[int] = None,      # Inteiro opcional
        image_ocid: Optional[str] = None,
        display_name: Optional[str] = None,
        ssh_authorized_keys_path: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create an OCI Compute instance via `oci` CLI. Missing parameters should be asked upstream by the agent.
    ## Example of expected parameters to create a compute instance: ##
    compartment-id: ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    subnet-id: ocid1.subnet.oc1.sa-saopaulo-1.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    shape: VM.Standard.E4.Flex
    availability-domain: IAfA:SA-SAOPAULO-1-AD-1
    image-id: ocid1.image.oc1.sa-saopaulo-1.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    display-name: teste_hoshikawa
    shape-config: '{"ocpus": 2, "memoryInGBs": 16}'
    """
    args = [
        "compute", "instance", "launch",
        "--compartment-id", compartment_ocid or "",
        "--subnet-id", subnet_ocid or "",
        "--shape", shape or "",
        "--shape-config", json.dumps({"ocpus": ocpus, "memoryInGBs": memory}),
        "--availability-domain", availability_domain or "",
        "--image-id", image_ocid or "",
        #"--source-details", json.dumps({"sourceType": "image", "imageId": image_ocid or ""}),
    ]
    if display_name:
        args += ["--display-name", display_name]
    if ssh_authorized_keys_path:
        args += ["--metadata", json.dumps({"ssh_authorized_keys": open(ssh_authorized_keys_path, "r", encoding="utf-8").read()})]
    if extra_args:
        args += extra_args

    # validate basics
    for flag in ["--compartment-id", "--subnet-id", "--shape", "--availability-domain"]:
        if "" in [args[args.index(flag)+1]]:
            return {"status": "error", "error": f"Missing required {flag} value"}

    code, out, err = oci_cli.run(args)
    if code != 0:
        return {"status": "error", "error": err.strip(), "stdout": out}
    try:
        payload = json.loads(out)
    except Exception:
        payload = {"raw": out}
    return {"status": "ok", "oci_result": payload}

@mcp.tool()
async def oci_cli_passthrough(raw: str) -> Dict[str, Any]:
    """Run an arbitrary `oci` CLI command (single string). Example: "network vcn list --compartment-id ocid1..."""
    args = shlex.split(raw)
    code, out, err = oci_cli.run(args)
    result = {"returncode": code, "stdout": out, "stderr": err}
    # try JSON parse
    try:
        result["json"] = json.loads(out)
    except Exception:
        pass
    return result

# -------------
# Entrypoint
# -------------
if __name__ == "__main__":
    # Start FastMCP server (stdio by default). A host (your agent/IDE) should launch this.
    mcp.run(transport="stdio")