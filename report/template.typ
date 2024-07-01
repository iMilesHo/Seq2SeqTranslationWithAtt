// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!

#let project(title: "", subTitle: "", authors: (), logo:"", date: "", body) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center)
  set text(font: "Times New Roman", lang: "en")
  set heading(numbering: "1.1.1")

  v(0.1fr)
  // Title page.
  // The page can contain a logo if you pass one with `logo: "logo.png"`.
  
  // Title row.
  align(center)[
    #block(text(weight: 700, 1.75em, title))
    #v(1em, weak: true)
    #block(text(weight: 700, 1.75em, subTitle))
    #v(1em, weak: true)
    #block(text(weight: 480, 1.2em, authors.map(a => a.name).at(0)))
    #v(0.3em, weak: true)
    #block(text(weight: 480, 1.2em, authors.map(a => a.studentNumber).at(0)))
    #v(0.3em, weak: true)
    #block(text(weight: 480, 1.2em, authors.map(a => a.email).at(0)))
  ]
  
  

  v(0.3fr)
  

  outline()
  v(0.3fr)
  pagebreak()
  // Main body.
  set par(justify: true)

  body
}