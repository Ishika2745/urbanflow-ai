import base64
from pathlib import Path

Path("assets").mkdir(exist_ok=True)

svg_bus = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg-bus" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#DC2626"/>
      <stop offset="100%" stop-color="#991B1B"/>
    </linearGradient>
  </defs>
  <rect width="400" height="240" rx="12" fill="url(#bg-bus)"/>
  <rect x="30" y="190" width="340" height="10" rx="5" fill="#7F1D1D"/>
  <rect x="50" y="60" width="300" height="120" rx="16" fill="#FFFFFF"/>
  <rect x="50" y="120" width="300" height="35" fill="#EF4444"/>
  <rect x="70" y="72" width="45" height="40" rx="4" fill="#93C5FD"/>
  <rect x="125" y="72" width="45" height="40" rx="4" fill="#93C5FD"/>
  <rect x="180" y="72" width="45" height="40" rx="4" fill="#93C5FD"/>
  <rect x="235" y="72" width="45" height="40" rx="4" fill="#93C5FD"/>
  <path d="M290 72 L335 85 L335 112 L290 112 Z" fill="#60A5FA"/>
  <rect x="140" y="46" width="120" height="18" rx="4" fill="#1E293B"/>
  <text x="200" y="59" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#FBBF24" text-anchor="middle">BEST MUMBAI</text>
  <circle cx="110" cy="180" r="22" fill="#1E293B"/>
  <circle cx="110" cy="180" r="10" fill="#94A3B8"/>
  <circle cx="280" cy="180" r="22" fill="#1E293B"/>
  <circle cx="280" cy="180" r="10" fill="#94A3B8"/>
  <circle cx="342" cy="145" r="6" fill="#FEF08A"/>
  <circle cx="58" cy="145" r="4" fill="#F87171"/>
</svg>'''

svg_rail = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg-rail" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2563EB"/>
      <stop offset="100%" stop-color="#1E3A8A"/>
    </linearGradient>
  </defs>
  <rect width="400" height="240" rx="12" fill="url(#bg-rail)"/>
  <line x1="20" y1="195" x2="380" y2="195" stroke="#64748B" stroke-width="4"/>
  <line x1="20" y1="205" x2="380" y2="205" stroke="#64748B" stroke-width="4"/>
  <path d="M80 60 C120 45, 280 45, 320 60 L335 175 L65 175 Z" fill="#FFFFFF"/>
  <path d="M95 70 C130 60, 270 60, 305 70 L315 120 L85 120 Z" fill="#1E293B"/>
  <path d="M105 75 C135 68, 265 68, 295 75 L302 110 L98 110 Z" fill="#38BDF8"/>
  <rect x="75" y="135" width="250" height="16" fill="#2563EB"/>
  <circle cx="120" cy="158" r="10" fill="#FEF08A" stroke="#CBD5E1" stroke-width="2"/>
  <circle cx="280" cy="158" r="10" fill="#FEF08A" stroke="#CBD5E1" stroke-width="2"/>
  <line x1="200" y1="45" x2="200" y2="25" stroke="#94A3B8" stroke-width="3"/>
  <line x1="170" y1="25" x2="230" y2="25" stroke="#E2E8F0" stroke-width="4"/>
  <text x="200" y="172" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#1E293B" text-anchor="middle">MUMBAI METRO &amp; RAIL</text>
</svg>'''

svg_walk = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg-walk" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#064E3B"/>
    </linearGradient>
  </defs>
  <rect width="400" height="240" rx="12" fill="url(#bg-walk)"/>
  <circle cx="200" cy="120" r="90" fill="none" stroke="rgba(255,255,255,0.2)" stroke-dasharray="6 4" stroke-width="2"/>
  <circle cx="200" cy="120" r="50" fill="none" stroke="rgba(255,255,255,0.35)" stroke-dasharray="6 4" stroke-width="2"/>
  <circle cx="200" cy="120" r="14" fill="#FBBF24" stroke="#FFFFFF" stroke-width="3"/>
  <path d="M80 190 L130 150 L200 120" stroke="#34D399" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M310 60 L260 90 L200 120" stroke="#34D399" stroke-width="4" fill="none" stroke-linecap="round"/>
  <circle cx="110" cy="135" r="8" fill="#FFFFFF"/>
  <path d="M110 144 L110 162 M102 150 L118 150 M110 162 L103 176 M110 162 L117 176" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
  <text x="200" y="32" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#A7F3D0" text-anchor="middle">400m (5 min) &amp; 800m (10 min) WALKING ACCESS</text>
</svg>'''

svg_ai = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bg-ai" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7C3AED"/>
      <stop offset="100%" stop-color="#4C1D95"/>
    </linearGradient>
  </defs>
  <rect width="400" height="240" rx="12" fill="url(#bg-ai)"/>
  <line x1="80" y1="70" x2="160" y2="130" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
  <line x1="160" y1="130" x2="240" y2="80" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
  <line x1="240" y1="80" x2="320" y2="150" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
  <line x1="160" y1="130" x2="210" y2="190" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
  <line x1="210" y1="190" x2="320" y2="150" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
  <circle cx="160" cy="130" r="20" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
  <circle cx="240" cy="80" r="16" fill="#38BDF8" stroke="#FFFFFF" stroke-width="2"/>
  <circle cx="80" cy="70" r="14" fill="#F59E0B" stroke="#FFFFFF" stroke-width="2"/>
  <circle cx="320" cy="150" r="20" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
  <circle cx="210" cy="190" r="14" fill="#F43F5E" stroke="#FFFFFF" stroke-width="2"/>
  <path d="M154 130 L158 134 L167 125" stroke="#FFFFFF" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M314 150 L318 154 L327 145" stroke="#FFFFFF" stroke-width="3" fill="none" stroke-linecap="round"/>
  <text x="200" y="32" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#DDD6FE" text-anchor="middle">NATURE-INSPIRED GENETIC &amp; PSO OPTIMIZATION</text>
</svg>'''

with open("assets/bus_network.svg", "w", encoding="utf-8") as f:
    f.write(svg_bus)
with open("assets/rail_metro.svg", "w", encoding="utf-8") as f:
    f.write(svg_rail)
with open("assets/walk_access.svg", "w", encoding="utf-8") as f:
    f.write(svg_walk)
with open("assets/ai_optimization.svg", "w", encoding="utf-8") as f:
    f.write(svg_ai)

print("4 local SVG graphics created successfully!")

