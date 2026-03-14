import json
from pathlib import Path

path = Path(r"d:\Cloud\OneDrive - Instituto Tecnológico de Celaya\Documentos\PhD-UW-Madison\Research\Chatbot1\ChatbotLP\notebooks\colab_supply_chain_chatbot_demo.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))
print('Total cells:', len(nb['cells']))
for i, cell in enumerate(nb['cells'][:25]):
    src = cell.get('source', [])
    first = src[0][:50] if src else ''
    print(i+1, cell['id'], cell['cell_type'], 'lines', len(src), 'first:', first)
