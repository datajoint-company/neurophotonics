# Neurophix

Neurophotonics simulation for probe optimization.


Be sure to add an environment file called `docker/dev.env` file with the following:

```shell
DJ_HOST={database host}
DJ_USER={username} 
DJ_PASS={password}
AWS_ACCESS_KEY={access id}
AWS_ACCESS_SECRET={access password}
DATABASE_PREFIX={prefix}
```

Fill the DJ fields in with the credentials of a user who is added to the neurophotonics database.
Fill the AWS fields with the AWS worker credentials found in lastpass

