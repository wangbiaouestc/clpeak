\NeedsTeXFormat{LaTeX2e}[2009/01/01]
\ProvidesClass{tu-thesis}[2010/09/09 v0.0 preliminary version]

% Default language is set to 'american'; to use 'ngerman', specify the
% 'ngerman' class option.  If you specify 'ngerman' *and* ('american' or
% 'british' or 'english'), the *first* option specified will be used for front
% matter and main text typesetting.  This is a contrary behaviour compared to
% the babel package.

\newif\if@english\@englishtrue
\newif\if@languageisset\@languageissetfalse
\def\tu@english{american}
\newenvironment{tuenglish}{\begin{otherlanguage}{american}}{\end{otherlanguage}}

\newcommand\tu@setenglish[1]{%
	\def\tu@english{#1}%
	\renewenvironment{tuenglish}{\begin{otherlanguage}{#1}}{\end{otherlanguage}}}

\def\tu@language{american}
\def\tu@activatelanguage{\selectlanguage{american}}

\DeclareOption{ngerman}{%
	\if@languageisset\else%
		\@englishfalse%
		\@languageissettrue%
		\def\tu@language{ngerman}%
	\fi}

\DeclareOption{english}{%
	\if@languageisset\else%
		\@englishtrue%
		\@languageissettrue%
		\def\tu@language{english}%
		\def\tu@activatelanguage{\selectlanguage{english}}%
	\fi
	\tu@setenglish{english}%
}

\DeclareOption{british}{%
	\if@languageisset\else%
		\@englishtrue%
		\@languageissettrue%
		\def\tu@language{british}%
		\def\tu@activatelanguage{\selectlanguage{british}}%
	\fi
	\tu@setenglish{british}%
}

\DeclareOption{american}{%
	\if@languageisset\else%
		\@englishtrue%
		\@languageissettrue%
		\def\tu@language{american}%
		\def\tu@activatelanguage{\selectlanguage{american}}%
	\fi
	\tu@setenglish{american}%
}


\DeclareOption{draft}{\def\tu@draft{true}}
\DeclareOption{final}{\def\tu@draft{false}}


\DeclareOption{bachelor}{\def\tu@thesisstyle{bachelor}}
\DeclareOption{master}{\def\tu@thesisstyle{master}}
\DeclareOption{diplom}{\def\tu@thesisstyle{diplom}}


% Pass all unknown options to IMTEKda
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{IMTEKda}}


%%% Default options

% Set English region to 'american', use 'ngerman' as main language
%\ExecuteOptions{american}

% Set non-draft typesetting as default
\ExecuteOptions{final}

% Set default thesis style to 'bachelor'
\ExecuteOptions{master}

% Process all options in the order specified
\ProcessOptions*

\if@english
	\PassOptionsToClass{englishpreamble,\tu@english}{IMTEKda}
	\PassOptionsToPackage{ngerman,\tu@english}{babel}
	\AtBeginDocument{\tu@activatelanguage}
	% TOFIX \tu@english is not working in case of american
	\RequirePackage[american]{babel}
\else
	\PassOptionsToClass{noenglishpreamble,ngerman}{IMTEKda}
	\PassOptionsToPackage{\tu@english,ngerman}{babel}
	\AtBeginDocument{\selectlanguage{ngerman}}
	% TOFIX ngerman is not working here
	\RequirePackage[german]{babel}
\fi


% To see the debug values, use this code snippet in a LaTeX document:
%	\makeatletter
%	\tu@debug
%	\makeatother
\def\tu@debug{%
	\fbox{\parbox{0.7\textwidth}{%
		\ttfamily
		\textbf{tu-thesis: Debug:} \\%
		$\backslash$if@languageisset: \if@languageisset Yes \else No \fi \\%
		$\backslash$tu@language:      \tu@language \\%
		$\backslash$if@english:       \if@english Yes \else No \fi \\%
		$\backslash$tu@english:       \tu@english
	}}
}

\AtEndOfClass{\tu@activatelanguage}

\LoadClass[\tu@thesisstyle,%
	   fontsize=12pt,%
	   twoside,%
	   paper=a4,%
	   titlepage,%
	   draft=\tu@draft,%
	   index=notnumbered,%
	   headsepline,%
	   toc_listof,%
	   tocbasic, %
	   listof=totoc,%
	   bibliography=totoc]{IMTEKda}

%%% \"Anderungen an IMTEKda.cls -- START

% Literatur-Verzeichnis
\if@english
\bibliographystyle{bib/latex8}
\else
\usepackage[comma,numbers,sort&compress]{natbib} % hypernat nach hyperref aktivieren!
\usepackage{bibgerm}
\bibliographystyle{gerplainnat}
\fi

\newcommand\acknowledgment{%
	\chapter*{\if@english
				Acknowledgment
			 \else 
			 	Danksagung
			\fi}	
}

\newcommand\MakeTheIMTEKdaVar[2]{%
	\newcommand{#1}{%
		\makeatletter%
		#2%
		\makeatother}%
}

\MakeTheIMTEKdaVar{\theuniversity}{\@university}
\MakeTheIMTEKdaVar{\thefaculty}{\@faculty}
\MakeTheIMTEKdaVar{\thechair}{\@chair}
\MakeTheIMTEKdaVar{\theinstitute}{\@institute}
\MakeTheIMTEKdaVar{\theaddress}{\@address}
\MakeTheIMTEKdaVar{\theplace}{\@place}
\MakeTheIMTEKdaVar{\thecourseofstudy}{\@courseofstudy}
\MakeTheIMTEKdaVar{\thedegree}{\@degree}
\MakeTheIMTEKdaVar{\thematricul}{\@matricul}
\MakeTheIMTEKdaVar{\thematriname}{\@matriname}
\MakeTheIMTEKdaVar{\thedpoversion}{\@dpoversion}
\MakeTheIMTEKdaVar{\thereferees}{\@referees}
\MakeTheIMTEKdaVar{\thesupervisor}{\@supervisor}
\MakeTheIMTEKdaVar{\thethesistime}{\@thesistime}

\def\university#1{\gdef\@university{#1}}
\def\@university{\@latex@error{No \noexpand\university given}\@ehc}
\def\faculty#1{\gdef\@faculty{#1}}
\def\@faculty{\@latex@error{No \noexpand\faculty given (e.g. Fakult\"at IV. Informatik und Elektrotechnik)}\@ehc}
\def\institute#1{\gdef\@institute{#1}}
\def\@institute{\@latex@error{No \noexpand\institute given (e.g. Institut f\"ur Technische Informatik und Mikroelektronik)}\@ehc}
\def\address#1{\gdef\@address{#1}}
\def\@address{\@latex@error{No \noexpand\address given (e.g. Franklinstra\ss{}e 28/29. D-10587 Berlin)}\@ehc}
\def\place#1{\gdef\@place{#1}}
\def\@place{\@latex@error{No \noexpand\place given (e.g. Berlin, Germany)}\@ehc}

\def\courseofstudy#1{\gdef\@courseofstudy{#1}}
\def\@courseofstudy{\@latex@error{No \noexpand\courseofstudy given}\@ehc}
\def\degree#1{\gdef\@degree{#1}}
\def\@degree{\@latex@error{No \noexpand\degree given}\@ehc}
\def\matricul#1{\gdef\@matricul{#1}}
\def\@matricul{\@latex@error{No \noexpand\matricul given}\@ehc}
\def\matriname#1{\gdef\@matriname{#1}}
\def\@matriname{\@latex@error{No \noexpand\matriname given}\@ehc}

\def\s@bject{%
  \if@diplom\if@english{Diploma Thesis}\else{Diplomarbeit}\fi%
  \else%
    \if@bachelor\if@english{Bachelor Thesis}\else{Bachelorarbeit}\fi%
    \else
    \if@english{Master Thesis}\else{Masterarbeit}\fi%
    \fi
  \fi}

\titlehead{%
	\includegraphics{./figures/TU_Logo_lang_RGB_rot}
}

    \renewcommand*\maketitle[1][1]{\begin{titlepage}%
% Seite groesser
    \enlargethispage{2cm}
    \setcounter{page}{#1}
    \let\footnotesize\small
    \let\footnoterule\relax
    \let\footnote\thanks
    \renewcommand*\thefootnote{\@fnsymbol\c@footnote}%
    \let\@oldmakefnmark\@makefnmark
    \renewcommand*{\@makefnmark}{\rlap\@oldmakefnmark}
    \ifx\@titlehead\@empty \else
        \noindent\begin{minipage}[t]{\textwidth}
        \raggedleft{
        \@titlehead
        }
        \end{minipage}\par
    \fi
    \null\vfill
    \begin{center}
    \vskip 3em
    \ifx\@subject\@empty \else
        {\Large \@subject \par}
        \vskip 3em
    \fbox{\parbox{\textwidth}{\begin{center} \vspace{0.5cm} 
    {{\titlefont \Large \@title }} \\ \vspace{0.5cm} 
    \Large \sffamily \@author\\ \vspace{0.5cm} 
    \sffamily \@matriname: \@matricul \vspace{0.5cm} \end{center}}}\\
    \vspace{1cm}
    \end{center}\par
    \vfill
\begin{flushleft}
	\sffamily
	\large
	\includegraphics[width=0.25\textwidth]{./figures/aes-logo}\\
	\@university\\
	\@faculty\\
	\@institute\\
	\@chair\\
	\@address\\
	\rmfamily
\end{flushleft}
    \vfill\null
    \if@twoside\next@tpage
        \noindent\begin{minipage}[t]{\textwidth}
        \@uppertitleback
        \end{minipage}\par
        \vfill
        \noindent\begin{minipage}[b]{\textwidth}
        \@lowertitleback
        \end{minipage}
    \fi
    \clearpage\thispagestyle{empty}
% Organisatorisches
    \noindent%
    % Breite der letzten Spalte bestimmen
    \setlength{\mytablewidth}{\textwidth}%
    \addtolength{\mytablewidth}{-4\tabcolsep}%
    \addtolength{\mytablewidth}{-0.24\textwidth}%
    % englische Variante
    \if@english
      \noindent
      \begin{tabular}[t]{p{0.24\textwidth}p{\mytablewidth}}
      &A thesis submitted in partial fulfillment of the requirements for the degree of
      \\[1em]
      &\@degree{} in \@courseofstudy{}
      \\[1em]
      &according to the examination regulations 
      at the \@university{}  for the Diploma in \@courseofstudy{} of
      \@dpoversion{}.\\[1em]
      &\@institute\\
      &\@chair\\
      &\@university\\
      &\@place
      \end{tabular}
      \vfil
      \noindent
      \begin{tabular}[t]{p{0.24\linewidth}p{\mytablewidth}}
      \bfseries Author&\begin{minipage}[t]{\mytablewidth}\@author\end{minipage}
      \end{tabular}
      \vfil
      \noindent
      \begin{tabular}[t]{p{0.24\linewidth}p{\mytablewidth}}
      \bfseries Thesis period&\begin{minipage}[t]{\mytablewidth}\@thesistime\end{minipage}\\&\\
      \bfseries Referees&\begin{minipage}[t]{\mytablewidth}\@referees\end{minipage}\\&\\
      \bfseries Supervisor&\begin{minipage}[t]{\mytablewidth}\@supervisor\end{minipage}\\
      \end{tabular}
      \clearpage\thispagestyle{empty}
      \null\vfill
      \noindent
      \begin{tabular}[t]{p{0.24\linewidth}p{\mytablewidth}}
        \bfseries\large Declaration&According to \S13(10) of the Examination
	Regulations\\[1em]
        &I hereby confirm to have written the following thesis on my own, not
	having used any other sources or resources than those listed.
        \\[1em]
        &\@place, \@date \\[2cm]
        &\begin{minipage}[t]{\mytablewidth}\@author\end{minipage}
      \end{tabular}
    % deutsche Variante
    \else
      \noindent%
      \begin{tabular}[t]{p{0.24\textwidth}p{\mytablewidth}}
      &Eingereichte Abschlu\ss{}arbeit gem\"a\ss{} den Bestimmungen der
      Pr\"ufungsordnung der \@university\ f\"ur den \@courseofstudy\ 
      vom \@dpoversion\\[1em]
      &\@institute\\
      &\@chair\\
      &\@university\\
      &\@place
      \end{tabular}
      \vfil
      \noindent
      \begin{tabular}[t]{p{0.24\linewidth}p{\mytablewidth}}
      \bfseries Autor&\begin{minipage}[t]{\mytablewidth}\@author\end{minipage}
      \end{tabular}
      \vfil
      \noindent
      \begin{tabular}[t]{p{0.24\linewidth}p{\mytablewidth}}
      \bfseries Bearbeitungszeit&\begin{minipage}[t]{\mytablewidth}\@thesistime\end{minipage}\\&\\
      \bfseries Gutachter&\begin{minipage}[t]{\mytablewidth}\@referees\end{minipage}\\&\\
      \bfseries Betreuer&\begin{minipage}[t]{\mytablewidth}\@supervisor\end{minipage}\\
      \end{tabular}
      \clearpage\thispagestyle{empty}
      \null\vfill
      \noindent
      \begin{tabular}[t]{p{0.24\linewidth}p{\mytablewidth}}
        \bfseries\large Erkl\"arung&Hiermit versichere ich an Eides statt, dass ich die vorliegende Arbeit
        selbst\"andig verfasst habe und keine anderen als die
        angegebenen Quellen und Hilfsmittel benutzt habe.\\[1em]
        &\@place, \@date \\[2cm]
        &\begin{minipage}[t]{\mytablewidth}\@author\end{minipage}
      \end{tabular}
	\fi
    \end{titlepage}
    \setcounter{footnote}{0}%
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\@extratitle\@empty
    \global\let\@titlehead\@empty
    \global\let\@subject\@empty
    \global\let\@publishers\@empty
    \global\let\@uppertitleback\@empty
    \global\let\@lowertitleback\@empty
    \global\let\@dedication\@empty
    \global\let\author\relax
    \global\let\title\relax
    \global\let\extratitle\relax
    \global\let\titlehead\relax
    \global\let\subject\relax
    \global\let\publishers\relax
    \global\let\uppertitleback\relax
    \global\let\lowertitleback\relax
    \global\let\dedication\relax
    \global\let\date\relax
    \global\let\and\relax
}




\university{Technische Universit\"at Berlin}
\address{Einsteinufer 17 $\cdot$ D-10587 Berlin}
\place{Berlin}




\if@english
\faculty{School IV $\cdot$ Electrical Engineering and Computer Science}
\institute{Department of Computer Engineering and Microelectronics}
\chair{Embedded Systems Architectures (AES)}
\matriname{Matriculation Number}
\else
\faculty{Fakult\"at IV $\cdot$ Informatik und Elektrotechnik}
\institute{Institut f\"ur Technische Informatik und Mikroelektronik (TIME)}
\chair{Fachgebiet Architektur Eingebetteter Systeme (AES)}
\matriname{Matrikelnummer}
\fi

\def\@abstractFileGerman{inc/abstract-de}
\def\@abstractFileEnglish{inc/abstract-en}

\def\@abstractErrorGerman{%
    \todo[inline]{Keine deutsche Zusammenfassung gefunden.}
    Das Kommando \texttt{$\backslash$abstracts} hat erfolglos nach der Datei
    \texttt{\@abstractFileGerman.tex} gesucht.  Damit die deutsche
    Zusammenfassung hier statt diesem nervigen Hinweis erscheint, reicht es
    aus, die Datei \texttt{\@abstractFileGerman.tex} zu erzeugen und dort den
    \LaTeX-Code beziehungsweise -Text einzuf\"ugen.}

\def\@abstractErrorEnglish{%
    \todo[inline]{No English abstract/summary found.}
    The command \texttt{$\backslash$abstracts} has searched for the file
    \texttt{\@abstractFileGerman.tex} unsuccessfully.  In order to let the
    English abstract show up here instead of this annoying note, it is
    sufficient to create a file \texttt{\@abstractFileGerman.tex} and place the
    according \LaTeX{} code and text, respectively.}

\newcommand\@abstractGerman{%
    \begin{abstract}%
    \IfFileExists{\@abstractFileGerman}{%
        \input{\@abstractFileGerman}}%
	{\@abstractErrorGerman}%
    \end{abstract}}

\newcommand\@abstractEnglish{%
    \begin{abstract}%
    \IfFileExists{\@abstractFileEnglish}{%
        \input{\@abstractFileEnglish}}%
	{\@abstractErrorEnglish}%
    \end{abstract}}

\newcommand\abstracts{%
	\if@english
		\tu@activatelanguage%
		\@abstractEnglish%
		\selectlanguage{ngerman}%
			\@abstractGerman%
		\tu@activatelanguage%
		
	\else
		\selectlanguage{ngerman}%
		\@abstractGerman%
		\tu@activatelanguage%
			\@abstractEnglish%
		\selectlanguage{ngerman}%
	\fi}
%%% \"Anderungen an IMTEKda.cls -- ENDE

% vim: set ft=tex ts=4 sw=4:
