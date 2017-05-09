import React from 'react'
import { config } from 'config'
import { rhythm } from 'utils/typography'
import profilePic from './profile-pic.jpg'

class Bio extends React.Component {
  render () {
    return (
      <p
        style={{
          marginBottom: rhythm(2.5),
        }}
      >
        <img
          src={profilePic}
          alt={`author ${config.authorName}`}
          style={{
            border: '1px solid #fff',
            boxShadow: '0 0 2px rgba(0, 0, 0, 0.125)',
            float: 'left',
            marginRight: rhythm(1/4),
            marginBottom: 0,
            width: rhythm(2),
            height: rhythm(2),
          }}
        />
        Written by <strong>{config.authorName}</strong> who lives and works in London building useful things. <a href="https://twitter.com/Piyush3dB">Find @Piyush3dB on Twitter.</a>
      </p>
    )
  }
}

export default Bio

