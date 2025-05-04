/* global L */

document.addEventListener("DOMContentLoaded", () => {
  // --- 1  MAP & BACKGROUND --------------------------------------------------
  const map = L.map("map", {
    center: [52.2297, 21.0122],
    zoom: 13,
    layers: []
  });

  const osm  = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { attribution: "&copy; OpenStreetMap contributors" }
  );
  const none = L.tileLayer("", { attribution: "" });

  L.control.layers({ OpenStreetMap: osm, "No background": none }, null, {
    position: "topright"
  }).addTo(map);

  none.addTo(map);        // start blank

  // --- 2  LOAD TRAM GRID ----------------------------------------------------
  const loading = L.control({ position: "topright" });
  loading.onAdd = () => {
    const box = L.DomUtil.create("div", "");
    box.style = "background:#fff;padding:10px;border-radius:5px;box-shadow:0 0 10px rgba(0,0,0,.2)";
    box.textContent = "Loading tram data…";
    return box;
  };
  loading.addTo(map);

  Promise.all([
    fetch("warsaw_tram_viz.geojson").then(r => r.json()),
    fetch("warsaw_trams.geojson").then(r => r.json())
  ])
    .then(([tramViz, rawTrams]) => {
      const tramLines = extractTramLines(rawTrams);
      visualizeGrid(tramViz, tramLines, map);
      map.removeControl(loading);
    })
    .catch(err => {
      console.error(err);
      map.removeControl(loading);
      alert("Could not load tram data. Please check that the files exist.");
    });
});

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function extractTramLines(tramData) {
  const lines = {};
  tramData.features.forEach(f => {
    if (f.geometry.type !== "LineString") return;
    const ref = f.properties.ref;
    if (!ref) return;                 // skip unnamed
    lines[ref] ||= {
      ref,
      name: f.properties.name || `Tram ${ref}`,
      color: colorFor(ref)
    };
  });
  return lines;
}

function visualizeGrid(tramViz, tramLines, map) {
  const outlineLayer = L.layerGroup().addTo(map);
  const whiteLayer   = L.layerGroup().addTo(map);
  const tramLayer    = L.layerGroup().addTo(map);
  const stopLayer    = L.layerGroup().addTo(map);

  const stroke = 5;  // Width of each tram line
  
  // Create a map of all endpoints to use for stop markers
  const endpoints = new Set();
  
  // Process each segment
  tramViz.features.forEach(feature => {
    if (feature.geometry.type !== "LineString") return;
    
    // Convert coordinates to Leaflet format
    const coords = feature.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
    if (coords.length < 2) return;
    
    // Get the lines that use this segment
    const lines = feature.properties.lines || [];
    if (!lines.length) return;
    
    // Calculate total width based on number of lines
    const segWidth = lines.length * (stroke + 1);
    
    // Add black outline
    L.polyline(coords, {
      color: "#000",
      weight: segWidth + 5,
      lineCap: "round",
      lineJoin: "round",
      smoothFactor: 1
    }).addTo(outlineLayer);
    
    // Add white background
    L.polyline(coords, {
      color: "#fff",
      weight: segWidth + 3,
      lineCap: "round",
      lineJoin: "round",
      smoothFactor: 1
    }).addTo(whiteLayer);
    
    // Add each colored line
    lines.forEach((line, idx) => {
      const lineColor = tramLines[line]?.color || colorFor(line.toString());
      
      // Calculate offset for this line
      // The key is to properly calculate the offset for each line
      const offset = idx * (stroke + 1) - (segWidth / 2) + ((stroke + 1) / 2);
      
      // Add the line with offset
      L.polyline(coords, {
        color: lineColor,
        weight: stroke,
        offset: offset,
        lineCap: "round",
        lineJoin: "round",
        smoothFactor: 1
      }).addTo(tramLayer);
    });
    
    // Track endpoints for stop markers
    if (coords.length > 0) {
      endpoints.add(JSON.stringify(coords[0]));
      endpoints.add(JSON.stringify(coords[coords.length - 1]));
    }
  });

  // Add stop markers at all endpoints
  Array.from(endpoints).forEach(pointStr => {
    const point = JSON.parse(pointStr);
    L.circleMarker(point, {
      radius: 15,
      color: "#000",
      weight: 3,
      fillColor: "#fff",
      fillOpacity: 1
    }).addTo(stopLayer);
  });

  // Fit map to bounds
  const allLatLng = tramViz.features.flatMap(f =>
    f.geometry.coordinates.map(([lon, lat]) => [lat, lon])
  );
  if (allLatLng.length) map.fitBounds(L.latLngBounds(allLatLng));

  // Add legend
  addLegend(map, Object.values(tramLines));
}

function addLegend(map, linesArr) {
  const legend = L.control({ position: "bottomright" });
  legend.onAdd = () => {
    const div = L.DomUtil.create("div", "info legend");
    Object.assign(div.style, {
      background: "#fff",
      padding: "10px",
      borderRadius: "5px",
      boxShadow: "0 0 15px rgba(0,0,0,.2)"
    });
    div.innerHTML = "<h4>Warsaw Tram Lines</h4>";

    linesArr
      .sort((a, b) => Number(a.ref) - Number(b.ref))
      .forEach(line => {
        div.innerHTML +=
          `<i style="background:${line.color};width:18px;height:18px;display:inline-block;margin-right:8px"></i>` +
          `Line ${line.ref}<br>`;
      });
    return div;
  };
  legend.addTo(map);
}

function colorFor(seed) {
  const palette = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b"
  ];
  
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = seed.charCodeAt(i) + ((hash << 5) - hash);
  }
  return palette[Math.abs(hash) % palette.length];
}