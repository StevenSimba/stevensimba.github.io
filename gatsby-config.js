//  $ npm run develop     $ gatsby develop  
// locahost:8000/   localhost:8000 / __graph1
// 

module.exports = {
  siteMetadata: {
    title: `Data Safari`,
    description: `Data, web and mobile apps`,
    author: `Steven Simba`,
    siteUrl: `https://stevensimba.github.io`,
    social: {
      twitter: `steven_simba`,
      facebook: ``,
      github: `stevensimba`,
      linkedin: `Steven_Simba`,
      email: `sigsimba@gmail.com`,
    },
  },
  plugins: [
  {
    resolve: `gatsby-plugin-google-analytics`,
    options: {
      trackingId: "UA-165984186-1",
      head: true,
    },

  },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        path: `${__dirname}/content/blog`,
        name: `blog`,
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        path: `${__dirname}/content/assets`,
        name: `assets`,
      },
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          {
            resolve: `gatsby-remark-images`,
            options: {
              maxWidth: 970,
            },
          },
          {
            resolve: `gatsby-remark-katex`, 
            options: {
              strict: `ignore`
            }
          },
          `gatsby-remark-prismjs`,
        ],
      },
    },
    `gatsby-transformer-sharp`,{

    resolve: `gatsby-plugin-sharp`,
    options: {
      useMozJpeg: true, 
      stripMetadata: true, 
      defaultQuality: 75,
    },
  },
    {
      resolve: `gatsby-plugin-google-analytics`,
      options: {
        //trackingId: `ADD YOUR TRACKING ID HERE`,
      },
    },
    `gatsby-plugin-feed`,
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: `flexible-gatsby-starter`,
        short_name: `flexible-gatsby`,
        start_url: `/`,
        background_color: `#663399`,
        theme_color: `#663399`,
        display: `minimal-ui`,
        icon: `./static/pngwave.png`, // This path is relative to the root of the site.
      },
    },
    // `gatsby-plugin-offline`,
    `gatsby-plugin-react-helmet`,
    `gatsby-plugin-sass`,
  ],
}
