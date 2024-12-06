import Navbar from './pages/Navbar'
import Upload from './pages/Upload'
import Verify from './pages/Verify'
import NewAnnotations from './pages/NewAnnotations'

import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { CssBaseline, ThemeProvider } from '@mui/material'

import theme from './theme'

function App() {
	return (
		<ThemeProvider theme={theme}>
			<CssBaseline />
			<Router>
				<Navbar />

				<Routes>
					<Route path='/' element={<Navigate to='/upload' />} />

					<Route path='/upload' element={<Upload />} />
					<Route path='/verify' element={<Verify />} />
					<Route path='/new-annotations' element={<NewAnnotations />} />
				</Routes>
			</Router>
		</ThemeProvider>
	)
}

export default App