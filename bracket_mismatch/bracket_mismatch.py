"""
Bracket mismatch helper for one line of text.

`remove_bracket_mismatch(line, rm=True)` returns the original line when the line
is valid and `None` on the first mismatch. With `rm=False`, it returns the first
offending bracket character and `None` when the line is balanced.
"""
BRACKETS = {"⧔":"⧕",
            "⦃":"⦄",
            "⍇":"⍈",
            "╔":"╗",
            "«":"»",
            "《":"》",
            "【":"】",
            "↣":"↢",
            "↪":"↩",
            "↬":"↫",
            "↱":"↰",
            "⇉":"⇇",
            "⇘":"⇙",
            "⇚":"⇛",
            "⇨":"⇦",
            "☾":"☽",
            "⇸":"⇷",
            "≼":"≽",
            "⋉":"⋊",
            "⋘":"⋙",
            "⋳":"⋻",
            "└":"┙",
            "┤":"├",
            "╕":"╒",
            "╢":"╟",
            "╣":"╠",
            "⟢":"⟣",
            "▙":"▟",
            "▛":"▜",
            "▞":"▚",
            "◨":"◧",
            "♞":"♘",
            "♚":"♔",
            "▧":"▨",
            "▶":"◀",
            "❪":"❫",
            "◂":"▸",
            "◃":"▹",
            "◅":"▻",
            "◑":"◐",
            "◖":"◗",
            "◢":"◣",
            "◮":"◭",
            "◳":"◰",
            "◶":"◵",
            "◷":"◴",
            "◺":"◿",
            "♡":"♥",
            "✠":"☩",
            "✦":"✱",
            "✪":"✬",
            "❰":"❱",
            "❮":"❯",
            "❴":"❵",
            "⟬":"⟭",
            "⟽":"⟾",
            "⦅":"⦆",
            "⦕":"⦖",
            "⦨":"⦩",
            "⧀":"⧁",
            "⩹":"⩺",
            "⪡":"⪢",
            "⪪":"⪫",
            "⬖":"⬗",
            "⮡":"⮠",
            "⮬":"⮭",
            "⮮":"⮯",
            "⯮":"⯬",
            "〖":"〗",
            "〘":"〙",
            "〔":"〕",
            "🠶":"🠴",
            "⦓":"⦔",
            "⯮":"⯬",
            "⋐":"⋑",
            "⤔":"⬹",
            "⇸":"⇷",
            "→":"←",
            "→":"←",
            "↠":"↞",
            "↱":"↰",
            "❮":"❯",
            "⦇":"⦈",
            "⦉":"⦊",
            "⇘":"⇙",
            "🢆":"🢇",
            "⎡":"⎤",
            "⦃":"⦄",
            "᚜":"᚛",
            "╘":"╛",
            "◟":"◞",
            "◜":"◝",
            "⥼":"⥽",
            "⥢":"⥤",
            "⋖":"⋗",
            "⮜":"⮞",
            "⫷":"⫸",
            "⟅":"⟆",
            "⤪":"⤨",
            "⨭":"⨮",
            "⸨":"⸩"
        }

def remove_bracket_mismatch(line, rm = True):
    s = []

    offset = 0
    while offset < len(line):
        if line[offset] in BRACKETS.keys():
            s.append(line[offset])
        elif line[offset] in BRACKETS.values():
            if s and BRACKETS[s[-1]] == line[offset]:
                s.pop()
            else:
                if rm:
                    return None
                else:
                    return line[offset]
        offset += 1
    if s:
        if rm:
            return None
        else:
            return s[-1]
    else:
        if rm:
            return line
        else:
            return None
