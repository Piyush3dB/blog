import React from 'react'
import moment from 'moment'
import Helmet from "react-helmet"
import ReadNext from '../components/ReadNext'
import { rhythm } from 'utils/typography'
import { config } from 'config'
import Bio from 'components/Bio'

//import '../css/prism-coy.css'
import '../css/zenburn.css'

class MarkdownWrapper extends React.Component {
  render () {
    const { route } = this.props
    const post = route.page.data

    return (
      <div className="markdown">
        <Helmet
          title={`${post.title} | ${config.blogTitle}`}
        />
        <h1 style={{marginTop: 0}}>{post.title}</h1>
        <div dangerouslySetInnerHTML={{ __html: post.body }} />
        <em
          style={{
            display: 'block',
            marginBottom: rhythm(2),
          }}
        >
          Posted {moment(post.date).format('MMMM D, YYYY')}
        </em>
        <hr
          style={{
            marginBottom: rhythm(2),
          }}
        />
        <ReadNext post={post} pages={route.pages} />
        <Bio />
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
          },
          extensions: ["color.js"]
        },
        tex2jax: {
          inlineMath: [ ['$','$'], ['\(', '\)'] ],
          displayMath: [ ['$$','$$'] ],
          processEscapes: true,
          useLabelIds: true,
          startNumber: 0
        },
        CommonHTML: {
          scale: 95
        }
      });
      MathJax.Hub.Queue(
        ["Typeset", MathJax.Hub],
        ["resetEquationNumbers",MathJax.InputJax.TeX],
        ["PreProcess",MathJax.Hub],
        ["Reprocess",MathJax.Hub]
      );
  }

} //class MarkdownWrapper

MarkdownWrapper.propTypes = {
  route: React.PropTypes.object,
}

export default MarkdownWrapper
