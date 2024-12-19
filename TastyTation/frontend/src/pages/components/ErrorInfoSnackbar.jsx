import { Snackbar, Alert } from '@mui/material'

export default function ErrorInfoSnackbar({ error, setError, info, infoMessage}) {
    return (
        <>
            {/* Error Snackbar */}
            {error && (
                <Snackbar
                    open={Boolean(error)}
                    anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
                >
                    <Alert severity='error' onClose={() => setError('')}>
                        {error}
                    </Alert>
                </Snackbar>
            )}
            {/* Verifying Status Snackbar */}
            {info && !error && (
                <Snackbar
                    open={info}
                    anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
                >
                    <Alert severity='info'>
                        {infoMessage}
                    </Alert>
                </Snackbar>
            )}
        </>
    )
}