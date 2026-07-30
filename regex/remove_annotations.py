"""
Remove annotation markers and most special symbols from
`TEXT15_Brackets_clean.txt` to create cleaner control text in
`TEXT15_Brackets_clean_ctrl.txt`.
"""
import re

with open("TEXT15_Brackets_clean.txt", "r", encoding="utf-8") as f:
    proc_content1 = f.read()

    # Removing all special characters
    spattern = re.compile(r"[#⧔⧕_⦃⦄\|~¤§«»÷•…‼⁄ↈ←→↞↠↢↣↩↪↫↬↰↱⇇⇉⇘⇙⇚⇛⇦⇨⇭⇷⇸≜≼≽⊏⊐⊛⊜⊢⊣⊤⊥⊪⊫⊯⊶⊷⋉⋊⋐⋑⋘⋙⋳⋻⌘⌺⍇⍈└┙├┤╒╓╔╕╗╘╙╚╛╜╝╠╣╟╢╤╦╩╪╫╬░▒▓▙▚▛▜▞▟■▣▧▨▭▯▶▷▸▹▻◀◁◂◃◅◈◉◍◎◐◑◔◖◗◙◢◣◤◥◧◨◩◪◬◭◮◰◱◲◳◴◵◶◷◸◹◺◿☆☩☽☾♔♖♘♚♜♞♠♡♢♣♥♦✙✜✝✞✠✢✤✦✧✪✬✭✱✺❃❉❋❑❪❫❮❯❰❱❴❵➠➨⟈⟉⟘⟙⟚⟛⟟⟢⟣⟬⟭⟴⟽⟾⤔⤨⤪⦅⦆⦓⦔⦕⦖⦨⦩⧀⧁⧋⧎⨭⨮⩩⪤◟◞◜◝⥼⥽⥢⥤⋖⋗⩹⩺⪪⪫⪡⪢⮜⮞⫷⫸⟅⟆⸨⸩⫞⫟⫠⫢⫣⫤⫥⫦⫧⫨⫩⫪⫫⫯⫰⫱⬖⬗⬹⭚⭛⭜⭞⭦⭧⭾⭿⮅⮈⮉⮋⮓⮔⮠⮡⮬⮭⯬⯮《》【】〔〕〖〗〘〙᚜᚛🢆🢇🞴🟂⛧⸘🟃◾◽❈⦇⦈⦉⦊⦻♂♀⚥⚩⥈⤂⤃⤡⤢⤤⤣⤥⤦]", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # Removing double spaces
    spattern = re.compile(r"  ", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, " ", proc_content1)

    # Removing line numbers
    #spattern = re.compile(r"^[^\n]{25}", flags=re.MULTILINE)
    #proc_content1 = re.sub(spattern, "", proc_content1)

    with open("TEXT15_Brackets_clean_ctrl.txt", "w", encoding="utf-8") as w:
        w.write(proc_content1)
