import React from 'react'
import moment from 'moment'
import { Link } from 'react-router'
import sortBy from 'lodash/sortBy'
import get from 'lodash/get'
import { prefixLink } from 'gatsby-helpers'
import { rhythm } from 'utils/typography'
import Helmet from "react-helmet"
import { config } from 'config'
import include from 'underscore.string/include'
import Bio from 'components/Bio'
//import {pathToDate} from 'utils'

class BlogIndex extends React.Component {
  render () {
    // Sort pages.
    const sortedPages = sortBy(this.props.route.pages, 'data.date')

    console.log(sortedPages);
    // Posts are those with md extension that are not 404 pages OR have a date (meaning they're a react component post).
    const visiblePages = sortedPages.filter(page => (
      get(page, 'file.ext') === 'md' && !include(page.path, '/404') || get(page, 'data.date')
    ))

    const pageLinks = visiblePages.map((page) => 
    (
        <li
          key={page.path}
          style={{
              marginBottom: rhythm(1/4),
          }}
        >
        <Link style={{boxShadow: 'none'}} to={prefixLink(page.path)}>
          {get(page, 'data.title', page.path)}
        </Link>
        <div style={{fontSize: 13, color: '#999', lineHeight: 1}}>
          {moment(page.date).format('MMMM D, YYYY')}
        </div>
      </li>
    ))

    return (
      <div>
        <Helmet
          title={config.blogTitle}
          meta={[
            {"name": "description", "content": "Sample blog"},
            {"name": "keywords", "content": "blog, articles"},
          ]}
        />
        <Bio />
        <ul>
          {pageLinks}
        </ul>
      </div>
    )
  }
}

BlogIndex.propTypes = {
  route: React.PropTypes.object,
}

export default BlogIndex
