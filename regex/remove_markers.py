"""
Clean annotated source text by removing headers, markers, and other artifacts,
then write the cleaned result to `TEXT15_Brackets.txt`.
"""
import re

with open("TTEXT15.txt", "r", encoding="utf-8") as f:
    proc_content1 = f.read()

    # At start of line markers, remove extra quotation
    spattern = re.compile(r"^\"", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # Fix the space between @@ and the paragraph number
    spattern = re.compile(r"@@ ", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "@@", proc_content1)

    # Fix quotation at the end of lines
    spattern = re.compile(r" +\"*\n", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "\n", proc_content1)

    # Replace comma replacement characters with actual commas again
    spattern = re.compile(r"[╪➠]", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, ",", proc_content1)

    # Take out any lines that are like headers, denoted by _§
    spattern = re.compile(r"^[^\n]+§[^\n]+\n", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # Take out tilde garbage ~~~~~
    spattern = re.compile(r"^[^\n]*~~[^\n]*\n", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # Delete blank lines 
    spattern = re.compile(r"^XXX[^a-z]+\): *\n", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # Fix extra space on hyphenated fractions
    spattern = re.compile(r"᚜ ", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "᚜", proc_content1)

    # Clean out the coreference patter "⮮XX⮯"
    spattern = re.compile(r"⮮[^⮮⮯\n]*⮯", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # Clean out markers
    spattern = re.compile(r"[#_\|¤§÷•…‼⁄ↈ⇭≜⊏⊐⊛⊜⊢⊣⊤⊥⊪⊫⊯⊶⊷⌘⌺╓╙╚╜╝╤╦╩╪╫╬░▒▓■▣▭▯▷◁◈◉◍◎◔◙◤◥◩◪◬◱◲◸◹☆♖♜♠♢♣♦✙✜✝✞✢✤✧✭✺❃❉❋❑➠➨⟈⟉⟘⟙⟚⟛⟟⟴⧋⧎⩩⪤≜◤◥⫞⫟⫠⫢⫣⫤⫥⫦⫧⫨⫩⫪⫫⫯⫰⫱⭚⭛⭜⭞⭦⭧⭾⭿⮅⮈⮉⮋⮓⮔◪🞴🟂✤⛧⸘🟃◾◽❈◎⧋⧎⦻~⦻♂♀⚥]", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # ''' Clean out every bracket, use this when you want to 
    #     take annotated text and need to get equivalent 
    #     raw text to feed into a control model'''
    # spattern = re.compile(r"[#⧔⧕_⦃⦄\|~¤§«»÷•…‼⁄ↈ←→↞↠↢↣↩↪↫↬↰↱⇇⇉⇘⇙⇚⇛⇦⇨⇭⇷⇸≜≼≽⊏⊐⊛⊜⊢⊣⊤⊥⊪⊫⊯⊶⊷⋉⋊⋐⋑⋘⋙⋳⋻⌘⌺⍇⍈└┙├┤╒╓╔╕╗╘╙╚╛╜╝╠╣╟╢╤╦╩╪╫╬░▒▓▙▚▛▜▞▟■▣▧▨▭▯▶▷▸▹▻◀◁◂◃◅◈◉◍◎◐◑◔◖◗◙◢◣◤◥◧◨◩◪◬◭◮◰◱◲◳◴◵◶◷◸◹◺◿☆☩☽☾♔♖♘♚♜♞♠♡♢♣♥♦✙✜✝✞✠✢✤✦✧✪✬✭✱✺❃❉❋❑❪❫❮❯❰❱❴❵➠➨⟈⟉⟘⟙⟚⟛⟟⟢⟣⟬⟭⟴⟽⟾⤔⤨⤪⦅⦆⦓⦔⦕⦖⦨⦩⧀⧁⧋⧎⨭⨮⩩⪤◟◞◜◝⥼⥽⥢⥤⋖⋗⩹⩺⪪⪫⪡⪢⮜⮞⫷⫸⟅⟆⸨⸩⫞⫟⫠⫢⫣⫤⫥⫦⫧⫨⫩⫪⫫⫯⫰⫱⬖⬗⬹⭚⭛⭜⭞⭦⭧⭾⭿⮅⮈⮉⮋⮓⮔⮠⮡⮬⮭⯬⯮《》【】〔〕〖〗〘〙᚜᚛🢆🢇🞴🟂⛧⸘🟃◾◽❈⦇⦈⦉⦊⦻♂♀⚥]", flags=re.MULTILINE)
    # proc_content1 = re.sub(spattern, "", proc_content1)

    # Change { because we want that to still generate its a bad marker and not yet in annotator
    spattern = re.compile(r"{", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "⦃", proc_content1)

    # Change } because we want that to still generate its a bad marker and not yet in annotator
    spattern = re.compile(r"}", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "⦄", proc_content1)

    # Change < because we want that to still generate its a bad marker and not yet in annotator
    spattern = re.compile(r"<", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "⧔", proc_content1)

    # Change > because we want that to still generate its a bad marker and not yet in annotator
    spattern = re.compile(r">", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "⧕", proc_content1)

    # Removing line numbers
    spattern = re.compile(r"^[^\n]{25}", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)

    # Removing ŁŁŁ
    spattern = re.compile(r"^[^\n]*ŁŁŁ[^\n]*\n", flags=re.MULTILINE)
    proc_content1 = re.sub(spattern, "", proc_content1)


    with open("TEXT15_Brackets.txt", "w", encoding="utf-8") as w:
        w.write(proc_content1)
