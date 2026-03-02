-- cell-markers.lua: Add data-cell-type to notebook code cell Divs
-- Quarto already sets id="cell-N" on code cell Divs (N = notebook cell index)
-- This filter adds data-cell-type="code" for explicit JS targeting

function Div(el)
  if el.classes:includes("cell") then
    el.attributes["data-cell-type"] = "code"
    return el
  end
end
