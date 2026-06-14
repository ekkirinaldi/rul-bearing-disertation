-- color-spans.lua
-- Convert Span elements with CSS color style to Word w:color markup.
-- Pandoc parses \textcolor{blue}{...} as Span with style="color: blue".
-- This filter rewrites each inline inside such a span, injecting an
-- OpenXML <w:rPr><w:color w:val="XXXXXX"/></w:rPr> fragment before each run.

local function hex_for_name(name)
  local map = {
    blue  = "0000FF",
    red   = "FF0000",
    green = "008000",
    black = "000000",
  }
  return map[name:lower()] or nil
end

local function extract_color(style)
  if not style then return nil end
  -- match: color: blue  or  color: #0000FF  or  color: #00F
  local name = style:match("color:%s*([a-zA-Z]+)")
  if name then return hex_for_name(name) end
  local hex6 = style:match("color:%s*#(%x%x%x%x%x%x)")
  if hex6 then return hex6:upper() end
  local hex3 = style:match("color:%s*#(%x%x%x)%f[^%x]")
  if hex3 then
    local r, g, b = hex3:sub(1,1), hex3:sub(2,2), hex3:sub(3,3)
    return (r..r..g..g..b..b):upper()
  end
  return nil
end

local function color_run_open(hex)
  return pandoc.RawInline("openxml",
    string.format('<w:rPr><w:color w:val="%s"/></w:rPr>', hex))
end

-- Wrap a list of inlines: prepend an openxml rPr snippet to trigger color.
-- We achieve this by wrapping each run in a Span that pandoc will emit as
-- a run with the matching rPr, but since pandoc ignores CSS color → w:color,
-- we instead use RawInline openxml fragments.
local function colorize_inlines(inlines, hex)
  local result = {}
  for _, inline in ipairs(inlines) do
    if inline.t == "Str" or inline.t == "Space" or inline.t == "SoftBreak" then
      -- wrap in a custom RawInline OOXML run
      local text = ""
      if inline.t == "Str" then
        text = inline.text:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
      elseif inline.t == "Space" or inline.t == "SoftBreak" then
        text = " "
      end
      table.insert(result, pandoc.RawInline("openxml",
        string.format('<w:r><w:rPr><w:color w:val="%s"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>',
          hex, text)))
    elseif inline.t == "Emph" then
      local inner = colorize_inlines(inline.content, hex)
      for _, item in ipairs(inner) do
        -- re-wrap with italic too
        if item.t == "RawInline" then
          local xml = item.text
          xml = xml:gsub('<w:rPr>', '<w:rPr><w:i/><w:iCs/>')
          table.insert(result, pandoc.RawInline("openxml", xml))
        else
          table.insert(result, item)
        end
      end
    elseif inline.t == "Strong" then
      local inner = colorize_inlines(inline.content, hex)
      for _, item in ipairs(inner) do
        if item.t == "RawInline" then
          local xml = item.text
          xml = xml:gsub('<w:rPr>', '<w:rPr><w:b/><w:bCs/>')
          table.insert(result, pandoc.RawInline("openxml", xml))
        else
          table.insert(result, item)
        end
      end
    elseif inline.t == "Span" then
      -- nested span: recurse, keeping color
      for _, item in ipairs(colorize_inlines(inline.content, hex)) do
        table.insert(result, item)
      end
    elseif inline.t == "RawInline" then
      table.insert(result, inline)
    elseif inline.t == "LineBreak" then
      table.insert(result, pandoc.RawInline("openxml", "<w:br/>"))
    elseif inline.t == "Code" then
      local text = inline.text:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
      table.insert(result, pandoc.RawInline("openxml",
        string.format('<w:r><w:rPr><w:color w:val="%s"/><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>',
          hex, text)))
    else
      -- fallback: keep as-is
      table.insert(result, inline)
    end
  end
  return result
end

function Span(el)
  local style = el.attr.attributes["style"] or ""
  local hex = extract_color(style)
  if not hex then return el end
  -- Replace span content with colorized raw OOXML runs
  return colorize_inlines(el.content, hex)
end
