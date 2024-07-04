# \definecolor{dkgreen}{rgb}{0,0.6,0}
# \definecolor{ltblue}{rgb}{0,0.4,0.4}
# \definecolor{dkviolet}{rgb}{0.3,0,0.5}

# %\def\lstlanguagefiles{defManSSR.tex}
# %\lstloadlanguages{SSR}
# \lstset{rangebeginprefix=(*\ begin\ ,%
#         rangebeginsuffix=\ *),%
#         rangeendprefix=(*\ end\ ,%
#         rangeendsuffix=\ *),
#         includerangemarker=false,
#         basicstyle=\ttfamily\small,
#         upquote=true}
# \lstdefinelanguage{Coq}{
#     % Anything betweeen $ becomes LaTeX math mode
#     mathescape=true,
#     % Comments may or not include Latex commands
#     texcl=false,
#     % Vernacular commands
#     morekeywords=[1]{Section, Module, End, Require, Import, Export,
#         Variable, Variables, Parameter, Parameters, Axiom, Hypothesis,
#         Hypotheses, Notation, Local, Tactic, Reserved, Scope, Open, Close,
#         Bind, Delimit, Definition, Let, Ltac, Fixpoint, CoFixpoint, Add,
#         Morphism, Relation, Implicit, Arguments, Unset, Contextual,
#         Strict, Prenex, Implicits, Inductive, CoInductive, Record,
#         Structure, Canonical, Coercion, Context, Class, Global, Instance,
#         Program, Infix, Theorem, Lemma, Corollary, Proposition, Fact,
#         Remark, Example, Proof, Goal, Save, Qed, Defined, Hint, Resolve,
#         Rewrite, View, Search, Show, Print, Printing, All, Eval, Check,
#         Projections, inside, outside, Def},
#     % Gallina
#     morekeywords=[2]{forall, exists, exists2, fun, fix, cofix, struct,
#         match, with, end, as, in, return, let, if, is, then, else, for, of,
#         nosimpl, when},
#     % Sorts
#     % Various tactics, some are std Coq subsumed by ssr, for the manual purpose
#     morekeywords=[4]{pose, set, move, case, elim, apply, clear, hnf,
#         intro, intros, generalize, rename, pattern, after, destruct,
#         induction, using, refine, inversion, injection, rewrite, congr,
#         unlock, compute, ring, field, fourier, replace, fold, unfold,
#         change, cutrewrite, simpl, have, suff, wlog, suffices, without,
#         loss, nat_norm, assert, cut, trivial, revert, bool_congr, nat_congr,
#         symmetry, transitivity, auto, split, left, right, autorewrite},
#     % Terminators
#     morekeywords=[5]{by, done, exact, reflexivity, tauto, romega, omega,
#         assumption, solve, contradiction, discriminate},
#     % Control
#     morekeywords=[6]{do, last, first, try, idtac, repeat},
#     % Comments delimiters, we do turn this off for the manual
#     morecomment=[s]{(*}{*)},
#     % Spaces are not displayed as a special character
#     showstringspaces=false,
#     % String delimiters
#     morestring=[b]",
#     morestring=[d]’,
#     % Size of tabulations
#     tabsize=3,
#     % Enables ASCII chars 128 to 255
#     extendedchars=false,
#     % Case sensitivity
#     sensitive=true,
#     % Automatic breaking of long lines
#     breaklines=false,
#     % Default style fors listings
#     basicstyle=\small,
#     % Position of captions is bottom
#     captionpos=b,
#     % flexible columns
#     columns=[l]flexible,
#     % Style for (listings') identifiers
#     identifierstyle={\ttfamily\color{black}},
#     % Style for declaration keywords
#     keywordstyle=[1]{\ttfamily\color{dkviolet}},
#     % Style for gallina keywords
#     keywordstyle=[2]{\ttfamily\color{dkgreen}},
#     % Style for sorts keywords
#     keywordstyle=[3]{\ttfamily\color{ltblue}},
#     % Style for tactics keywords
#     keywordstyle=[4]{\ttfamily\color{dkblue}},
#     % Style for terminators keywords
#     keywordstyle=[5]{\ttfamily\color{dkred}},
#     %Style for iterators
#     %keywordstyle=[6]{\ttfamily\color{dkpink}},
#     % Style for strings
#     stringstyle=\ttfamily,
#     % Style for comments
#     commentstyle={\ttfamily\color{dkgreen}},
#     %moredelim=**[is][\ttfamily\color{red}]{/&}{&/},
#     literate=
#     {\[\[}{$\llbracket$}1
#     {\]\]}{$\rrbracket$}1
#     {\\em}{$\emptyset$}1
#     {\\.}{$\cdot$}1
#     {\\forall}{{\color{dkgreen}{$\forall\;$}}}1
#     {\\exists}{{$\exists\;$}}1
#     {<-}{{$\leftarrow\;$}}1
#     {=>}{{$\Rightarrow\;$}}1
#     {==}{{\code{==}\;}}1
#     {==>}{{\code{==>}\;}}1
#     %    {:>}{{\code{:>}\;}}1
#     {->}{{$\rightarrow\;$}}1
#     {<->}{{$\leftrightarrow\;$}}1
#     {<==}{{$\leq\;$}}1
#     {\#}{{$^\star$}}1
#     {\\o}{{$\circ\;$}}1
#     {\@}{{$\cdot$}}1
#     {\/\\}{{$\wedge\;$}}1
#     {\\\/}{{$\vee\;$}}1
#     {++}{{\code{++}}}1
#     {~}{{$\sim$}}1
#     {\@\@}{{$@$}}1
#     {\\mapsto}{{$\mapsto\;$}}1
#     {\\hline}{{\rule{\linewidth}{0.5pt}}}1
#     %
# }[keywords,comments,strings]
# \lstnewenvironment{coqcode}{\lstset{language=Coq,mathescape=true,basicstyle=\ttfamily\small}}{}
# \newcommand{\CC}{\lstinline[language=Coq, basicstyle=\ttfamily\small]}

# \lstnewenvironment{hask}{\lstset{language=Haskell,mathescape=true,basicstyle=\ttfamily\small}}{}
# \newcommand{\HC}{\lstinline[language=Haskell, basicstyle=\ttfamily\small]}


# This is a renderer for Coq code.

from PIL import Image, ImageDraw, ImageFont


vernacs = [
    "Section",
    "Module",
    "End",
    "Require",
    "Import",
    "Export",
    "Variable",
    "Variables",
    "Parameter",
    "Parameters",
    "Axiom",
    "Hypothesis",
    "Hypotheses",
    "Notation",
    "Local",
    "Tactic",
    "Reserved",
    "Scope",
    "Open",
    "Close",
    "Bind",
    "Delimit",
    "Definition",
    "Let",
    "Ltac",
    "Fixpoint",
    "CoFixpoint",
    "Add",
    "Morphism",
    "Relation",
    "Implicit",
    "Arguments",
    "Unset",
    "Contextual",
    "Strict",
    "Prenex",
    "Implicits",
    "Inductive",
    "CoInductive",
    "Record",
    "Structure",
    "Canonical",
    "Coercion",
    "Context",
    "Class",
    "Global",
    "Instance",
    "Program",
    "Infix",
    "Theorem",
    "Lemma",
    "Corollary",
    "Proposition",
    "Fact",
    "Remark",
    "Example",
    "Proof",
    "Goal",
    "Save",
    "Qed",
    "Defined",
    "Hint",
    "Resolve",
    "Rewrite",
    "View",
    "Search",
    "Show",
    "Print",
    "Printing",
    "All",
    "Eval",
    "Check",
    "Projections",
    "inside",
    "outside",
    "Def",
]

gallina = [
    "forall",
    "exists",
    "exists2",
    "fun",
    "fix",
    "cofix",
    "struct",
    "match",
    "with",
    "end",
    "as",
    "in",
    "return",
    "let",
    "if",
    "is",
    "then",
    "else",
    "for",
    "of",
    "nosimpl",
    "when",
]

sorts = ["Type", "Prop", "Set", "true", "false", "option"]

colors = {
    "dkgreen": (0, int(0.6 * 255), 0),
    "ltgreen": (200, 255, 200),
    "bggreen": (200, 255, 200),

    "dkblue": (0, 0, 255),
    "ltblue": (0, int(0.4 * 255), int(0.4 * 255)),
    "bgblue": (220, 220, 255),
    
    "dkviolet": (int(0.3 * 255), 0, int(0.5 * 255)),
    "ltviolet": (255, 200, 255),
    "bgviolet": (255, 200, 255),
    
    "dkpurple": (int(0.5 * 255), 0, int(0.3 * 255)),
    "ltpurple": (255, 220, 255),
    "bgpurple": (255, 220, 255),
    
    "dkred": (int(0.5 * 255), 0, 0),
    "ltred": (255, 210, 210),
    "bgred": (255, 210, 210),
    
    "black": (0, 0, 0),
}

ligatures = {
    "forall": "∀",
    "exists": "∃",
    "=>": "⇒",
    "<-": "←",
    "->": "→",
    "<->": "↔",
    "==": "==",
    "===>": "==>",
}

def tokenize(code):
    # A token is either a string or a whitespace character
    tokens = []
    current_token = ""
    for char in code:
        if char in [" ", "\n", "\t"]:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char 
    if current_token:
        tokens.append(current_token)
    return tokens



def line_widths(elements):
    lines = {}
    for element in elements:
        if element.get("line") is None:
            continue

        if element["line"] not in lines:
            lines[element["line"]] = 0
        if element["type"] == "text":
            lines[element["line"]] += element["width"]
    return lines.values()


def render_coq(code, font_size, line_numbers, markers, output_file):
    fonts = {
        # "N": ImageFont.truetype("Libertine/LinLibertine_R.ttf", font_size),
        # "B": ImageFont.truetype("Libertine/LinLibertine_RB.ttf", font_size),
        # "I": ImageFont.truetype("Libertine/LinLibertine_RI.ttf", font_size),
        'N': ImageFont.truetype("Inconsolata/static/Inconsolata-Regular.ttf", font_size),
        'B': ImageFont.truetype("Inconsolata/static/Inconsolata-Bold.ttf", font_size),
        # 'I': ImageFont.truetype("/Users/akeles/Programming/projects/PbtBenchmark/coderenderer/Inconsolata/static/Incosolata-Italic.ttf", font_size),
    }

    for literal, ligature in ligatures.items():
        code = code.replace(literal, ligature)

    code = code.strip()
    elements = []

    for marker in markers:
        elements.append(
            {
                "type": "line_based_rounded_rectangle",
                "lines": marker["lines"],
                "fill": colors["bg" + marker["color"]],
                "outline": colors["dk" + marker["color"]],
                "width": 3,
                "radius": 5,
                "z": 0,
            }
        )

    CODE_START_AT = 10
    LINE_HEIGHT = font_size * 6 // 5
    x = CODE_START_AT
    y = 10
    incomment = False
    line = 1

    if line_numbers:
        CODE_START_AT = 50
        x = CODE_START_AT

        for i in range(1, code.count("\n") + 2):
            elements.append(
                {
                    "type": "text",
                    "x": 10,
                    "y": y,
                    "width": LINE_HEIGHT,
                    "height": LINE_HEIGHT,
                    "color": colors["black"],
                    "font": fonts["N"],
                    "text": str(i),
                    "line": None,
                    "z": 1,
                }
            )
            y += LINE_HEIGHT
        y = 10

    for token in tokenize(code):
        if token == "(*":
            incomment = True

        if incomment:
            font = fonts["N"]
            color = colors["dkgreen"]
        elif token in vernacs:
            font = fonts["B"]
            color = colors["dkviolet"]
        elif token in gallina:
            font = fonts["N"]
            color = colors["dkgreen"]
        elif token in sorts:
            font = fonts["N"]
            color = colors["ltblue"]
        else:
            font = fonts["N"]
            color = colors["black"]

        if token == "*)":
            incomment = False

        if token == "\n":
            y += LINE_HEIGHT
            line += 1
            x = CODE_START_AT
        else:
            token = token
            bbox = font.getbbox(token)
            txt_width = bbox[2] - bbox[0]
            # draw.text((x, y), token, font=font, fill=color)
            elements.append(
                {
                    "type": "text",
                    "x": x,
                    "y": y,
                    "width": txt_width,
                    "height": LINE_HEIGHT,
                    "color": color,
                    "font": font,
                    "text": token,
                    "line": line,
                    "z": 1,
                }
            )
            x += txt_width

    elements = sorted(elements, key=lambda x: x["z"])

    # Height is LINE_HEIGHT per line, plus 20 for padding
    height = (1 + code.count("\n")) * LINE_HEIGHT + 20
    # Width is the width of the longest line, plus 20 for padding, and 50 if line numbers are enabled
    width = max(line_widths(elements)) + CODE_START_AT + (LINE_HEIGHT if line_numbers else 20)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for element in elements:
        match element["type"]:
            case "text":
                draw.text(
                    (element["x"], element["y"]),
                    element["text"],
                    font=element["font"],
                    fill=element["color"],
                )
            case "rectangle":
                draw.rectangle(
                    element["coords"],
                    fill=element["fill"],
                    outline=element["outline"],
                    width=element["width"],
                )
            case "absolute_rounded_rectangle":
                draw.rounded_rectangle(
                    element["coords"],
                    fill=element["fill"],
                    outline=element["outline"],
                    width=element["width"],
                    radius=element["radius"],
                )
            case "line_based_rounded_rectangle":
                # Find the first and last non-whitespace token in all lines, take the max and min x
                left = 999999
                right = 0
                for line in range(element["lines"][0], element["lines"][1] + 1):
                    line_tokens = list(filter(lambda x: x["type"] == "text" and x["line"] == line, elements))
                    line_tokens = list(filter(lambda x: x["text"] not in ["\t", " ", "\n"], line_tokens))
                    line_tokens = sorted(line_tokens, key=lambda x: x["x"])
                    print(line)
                    first_token = line_tokens[0]
                    last_token = line_tokens[-1]
                    left = min(left, first_token["x"])
                    right = max(right, last_token["x"] + last_token["width"])

                top = LINE_HEIGHT * element["lines"][0] - LINE_HEIGHT * 5 // 6
                bottom = LINE_HEIGHT * element["lines"][1]  + LINE_HEIGHT // 4

                draw.rounded_rectangle(
                    [left - 10, top, right + 10, bottom],
                    fill=element["fill"],
                    outline=element["outline"],
                    width=element["width"],
                    radius=element["radius"],
                )

    img.save(output_file)


code = """
Definition runLoop (fuel : nat) (cprop : CProp ∅): G Result :=  
  let fix runLoop'
    (fuel : nat) 
    (cprop : CProp ∅) 
    (passed : nat)
    (discards: nat)
    : G Result :=
    match fuel with
    | O => ret (mkResult discards false passed [])
    | S fuel' => 
        res <- genAndRun cprop (log2 (passed + discards));;
        match res with
        | Normal seed false =>
            (* Fails *)
            let shrinkingResult := shrinkLoop 10 cprop seed in
            let printingResult := print cprop 0 shrinkingResult in
            ret (mkResult discards true (passed + 1) printingResult)
        | Normal _ true =>
            (* Passes *)
            runLoop' fuel' cprop (passed + 1) discards
        | Discard _ _ => 
            (* Discard *)
            runLoop' fuel' cprop passed (discards + 1)
        end
    end in
    runLoop' fuel cprop 0 0.
"""

markers = [
    { "lines": (9, 9), "color": "green" },
    { "lines": (11, 11), "color": "purple" },
    { "lines": (14, 17), "color": "red" },
    { "lines": (19, 20), "color": "blue" },
    { "lines": (22, 23), "color": "blue" },
]

render_coq(code, 40, True, markers, "runLoop.png")




code = """
Definition targetLoop (fuel : nat) (cprop : CProp ∅)
  (feedback_function: ⟦⦗cprop⦘⟧ -> Z) {Pool : Type}
  {poolType: @SeedPool (⟦⦗cprop⦘⟧) Z Pool} (seeds : Pool)
  (utility: Utility) : G Result :=
  let fix targetLoop' 
         (fuel : nat) (passed : nat) (discards: nat)
         {Pool : Type} (seeds : Pool)
         (poolType: @SeedPool (⟦⦗cprop⦘⟧) Z Pool)
         (utility: Utility) : G Result :=
        match fuel with
        | O => ret (mkResult discards false passed [])
        | S fuel' => 
            let directive := sample seeds in
            res <- genAndRunWithDirective cprop directive (Nat.log2 (passed + discards)%nat);;
            match res with
            | Normal seed false =>
                (* Fails *)
                let shrinkingResult := shrinkLoop 10 cprop seed in
                let printingResult := print cprop 0 shrinkingResult in
                ret (mkResult discards true (passed + 1) printingResult)
            | Normal seed true =>
                (* Passes *)
                let feedback := feedback_function seed in
                match useful seeds feedback with
                | true =>
                    let seeds' := invest (seed, feedback) seeds in
                    targetLoop' fuel' (passed + 1) discards seeds' poolType utility
                | false =>
                    let seeds' := match directive with
                                  | Generate => seeds
                                  | Mutate source => revise seeds
                                  end in
                    targetLoop' fuel' (passed + 1) discards seeds' poolType utility
                end
            | Discard _ _ => 
                (* Discard *)
                targetLoop' fuel' passed (discards + 1) seeds poolType utility
            end
        end in
        targetLoop' fuel 0 0 seeds poolType utility.
"""

markers = [
    { "lines": (11, 11), "color": "green" },
    { "lines": (14, 14), "color": "purple" },
    { "lines": (17, 20), "color": "red" },
    { "lines": (22, 33), "color": "blue" },
    { "lines": (36, 37), "color": "blue" },
]

render_coq(code, 40, True, markers, "targetLoop.png")

