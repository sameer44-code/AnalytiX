import os
import io
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader
import networkx as nx

def parse_uploaded_file(file_path: str, filename: str) -> tuple[pd.DataFrame, str]:
    """
    Parses an uploaded file into a pandas DataFrame.
    Supported extensions: .csv, .xlsx, .xls, .html, .htm, .pdf, .dot, .gv
    Returns: (df, detected_type)
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".csv":
        # Handle potential encoding issues
        for encoding in ["utf-8", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df, "csv"
            except UnicodeDecodeError:
                continue
            except Exception:
                pass
        df = pd.read_csv(file_path, on_bad_lines="skip")
        return df, "csv"

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        return df, "xlsx"

    elif ext in [".html", ".htm"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, "html.parser")
        tables = soup.find_all("table")
        
        if tables:
            # First try direct extraction via BeautifulSoup
            for tbl in tables:
                rows = []
                headers = []
                th_tags = tbl.find_all("th")
                if th_tags:
                    headers = [th.get_text().strip() for th in th_tags]
                
                tr_tags = tbl.find_all("tr")
                for tr in tr_tags:
                    td_tags = tr.find_all("td")
                    if td_tags:
                        row_vals = [td.get_text().strip() for td in td_tags]
                        if not headers:
                            headers = [f"Col_{i+1}" for i in range(len(row_vals))]
                        if len(row_vals) == len(headers):
                            rows.append(row_vals)
                if rows:
                    return pd.DataFrame(rows, columns=headers), "html"

            # Fallback to pandas read_html
            try:
                dfs = pd.read_html(io.StringIO(html_content))
                if dfs:
                    df = max(dfs, key=lambda x: x.shape[0] * x.shape[1])
                    return df, "html"
            except Exception:
                pass
        
        # Fallback if no table tags: parse text lines into structured rows
        lines = [line.strip() for line in soup.get_text().splitlines() if line.strip()]
        df = pd.DataFrame({"Text_Content": lines[:1000]})
        return df, "html"

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        extracted_rows = []
        headers = None
        
        for page in reader.pages:
            text = page.extract_text() or ""
            lines = text.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Split by multiple spaces or commas/tabs
                parts = re.split(r'\s{2,}|\t|,', line)
                if len(parts) >= 2:
                    if headers is None:
                        headers = [f"Col_{i}_{p[:12].strip()}" for i, p in enumerate(parts)]
                    else:
                        if len(parts) == len(headers):
                            extracted_rows.append(parts)
                        elif len(parts) > len(headers):
                            extracted_rows.append(parts[:len(headers)])
                        else:
                            # Pad with None
                            extracted_rows.append(parts + [None] * (len(headers) - len(parts)))
        
        if extracted_rows and headers:
            df = pd.DataFrame(extracted_rows, columns=headers)
        else:
            # Fallback: line by line content
            all_text_lines = []
            for page in reader.pages:
                t = page.extract_text() or ""
                all_text_lines.extend([l.strip() for l in t.splitlines() if l.strip()])
            df = pd.DataFrame({"Document_Line": all_text_lines[:2000]})
        
        return df, "pdf"

    elif ext in [".dot", ".gv"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            dot_content = f.read()
        
        # Parse graph edges and nodes from DOT format
        nodes = set()
        edges = []
        
        # Match edge patterns like: "nodeA" -> "nodeB" [weight=5] or nodeA -- nodeB
        edge_pattern = re.findall(r'([\"\'\w]+)\s*(?:->|--)\s*([\"\'\w]+)(?:\s*\[([^\]]*)\])?', dot_content)
        for src, dst, attrs in edge_pattern:
            src = src.strip('"\' ')
            dst = dst.strip('"\' ')
            nodes.add(src)
            nodes.add(dst)
            weight = 1.0
            if attrs and "weight" in attrs:
                m = re.search(r'weight\s*=\s*([0-9\.]+)', attrs)
                if m:
                    weight = float(m.group(1))
            edges.append({
                "Source_Node": src,
                "Target_Node": dst,
                "Weight": weight,
                "Attributes": attrs.strip() if attrs else "default"
            })
            
        if edges:
            df = pd.DataFrame(edges)
        else:
            # Look for node definitions
            node_pattern = re.findall(r'([\"\'\w]+)\s*\[([^\]]*)\]', dot_content)
            node_rows = []
            for n, attrs in node_pattern:
                node_rows.append({
                    "Node_ID": n.strip('"\' '),
                    "Attributes": attrs.strip()
                })
            df = pd.DataFrame(node_rows) if node_rows else pd.DataFrame({"Raw_DOT": dot_content.splitlines()[:1000]})
            
        return df, "dot"

    else:
        # Default fallback to CSV reader
        df = pd.read_csv(file_path)
        return df, "csv"
