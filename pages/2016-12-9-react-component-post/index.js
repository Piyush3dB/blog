const React = require('react')
const DatePicker = require('./single-date-picker')
require('react-dates/css/variables.scss')
require('react-dates/css/styles.scss')
var md = require('markdown-it')()
            .use(require('markdown-it-mathjax')());

class Post extends React.Component {
  render () {

    return (
      <div>
        <h1>{this.props.route.page.data.title}</h1>
        <p>Word to the javascript yos</p>
        <p>This is the best I think</p>
        <p>Cause you can now do stuff like... embed a date picker in your blog posts!</p>
        <DatePicker />
        <br />
        <br />
        <p>(No doubt a secret dream of yours)</p>
      </div>
    )
  }


  componentDidMount()  { this.queueToMathjaxHub(); }
  componentDidUpdate() { this.queueToMathjaxHub(); }

  queueToMathjaxHub(){
      MathJax.Hub.Config({
        TeX: {
          equationNumbers: {
            autoNumber: "AMS"
          }
        },
        tex2jax: {
          inlineMath: [ ['$','$'] ],
          displayMath: [ ['$$','$$'] ],
          processEscapes: true,
          useLabelIds: true,
          startNumber: 0
        }
      });
      MathJax.Hub.Queue(
        ["Typeset", MathJax.Hub],
        ["resetEquationNumbers",MathJax.InputJax.TeX],
        ["PreProcess",MathJax.Hub],
        ["Reprocess",MathJax.Hub]
      );
  }


}

export default Post

exports.data = {
  title: "A post written in Javascript!",
  date: "2016-12-09T12:40:32.169Z",
}
